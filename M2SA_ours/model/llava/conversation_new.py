import dataclasses
from enum import Enum, auto
from typing import List, Tuple, Optional, Dict


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False
    
    # New fields for part-whole segmentation
    enable_part_whole: bool = False
    part_whole_prompt: str = ""

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if "mmtag" in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0:
                        message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def get_part_whole_prompt(self, user_description: str) -> str:
        """Generate prompt for part-whole decomposition."""
        if not self.enable_part_whole:
            return user_description
            
        if not self.part_whole_prompt:
            # Default part-whole prompt
            prompt = f"""Please analyze the following description and determine if it refers to a whole object or a part of an object. Then provide two descriptions:
                        1. Whole description: A description of the complete object
                        2. Part description: A description of the specific part (if applicable)

                            Input description: "{user_description}"

                            Please respond in the following format:
                                Whole: [whole object description]
                                Part: [part description, or "N/A" if the input describes a whole object]
                     Analysis:,
                      """
        else:
            prompt = self.part_whole_prompt.format(sentence=user_description)
        
        return prompt

    def parse_part_whole_response(self, response: str) -> Tuple[str, Optional[str]]:
        """Parse the response from part-whole decomposition."""
        lines = response.strip().split('\n')
        whole_desc = None
        part_desc = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Whole:'):
                whole_desc = line[6:].strip()
            elif line.startswith('Part:'):
                part_desc = line[5:].strip()
                if part_desc.lower() in ['n/a', 'na', 'none', '']:
                    part_desc = None
        
        return whole_desc , part_desc

    def process_segmentation_input(self, user_description: str, llava_model=None) -> Tuple[str, Optional[str], bool]:
        """
        Process user input for segmentation task.
        Returns: (whole_description, part_description, is_part_whole)
        """
        if not self.enable_part_whole or not llava_model:
            return user_description, None, False
        
        # Generate part-whole analysis prompt
        analysis_prompt = self.get_part_whole_prompt(user_description)
        
        # Use LLaVA to analyze the description
        # This is a placeholder - you'll need to implement the actual LLaVA inference
        try:
            analysis_response = self._call_llava_for_analysis(analysis_prompt, llava_model)
            whole_desc, part_desc = self.parse_part_whole_response(analysis_response)
            
            # If part description exists, it's a part-whole scenario
            is_part_whole = part_desc is not None
            
            return whole_desc, part_desc, is_part_whole
            
        except Exception as e:
            print(f"Error in part-whole analysis: {e}")
            return user_description, None, False

    def _call_llava_for_analysis(self, prompt: str, llava_model) -> str:
        return 

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO

                    from PIL import Image

                    msg, image, image_process_mode = msg
                    if image_process_mode == "Pad":

                        def expand2square(pil_img, background_color=(122, 116, 104)):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(
                                    pil_img.mode, (width, width), background_color
                                )
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(
                                    pil_img.mode, (height, height), background_color
                                )
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result

                        image = expand2square(image)
                    elif image_process_mode == "Crop":
                        pass
                    elif image_process_mode == "Resize":
                        image = image.resize((336, 336))
                    else:
                        raise ValueError(
                            f"Invalid image_process_mode: {image_process_mode}"
                        )
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    if return_pil:
                        images.append(image)
                    else:
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO

                    msg, image, image_process_mode = msg
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    ret.append([img_str, None])
                    msg = msg.replace("<image>", "").strip()
                    if len(msg) > 0:
                        ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
            enable_part_whole=self.enable_part_whole
        )

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [
                    [x, y[0] if type(y) is tuple else y] for x, y in self.messages
                ],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


# Original conversation templates
conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        (
            "Human",
            "What are the key differences between renewable and non-renewable energy sources?",
        ),
        (
            "Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n",
        ),
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(("Human", "Hi!"), ("Assistant", "Hi there! How can I help you today?")),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)

conv_chatml = Conversation(
        system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant name StableCapybara. The assistant gives helpful and honest answers.""",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.TWO,
        sep="###",
        sep2="###",
        messages=(),
        offset=0,
        #stop_token_ids=[50278, 50279, 50277, 1, 0],
)

# New conversation template for part-whole segmentation
conv_llava_part_whole = Conversation(
    system="You are a helpful vision assistant specializing in image segmentation. "
    "You can analyze descriptions to determine if they refer to whole objects or parts of objects, "
    "and provide accurate segmentation guidance.",
    roles=("USER", "ASSISTANT"),
    version="part_whole_v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    enable_part_whole=True,
    part_whole_prompt="""Analyze the following description: "{sentence}"

    Please determine if this description refers to:
    1. A complete/whole object
    2. A part or component of a larger object

    If it's a part, please provide both the whole object description and the specific part description.

    Respond in this exact format:
    Whole: [description of the complete object]
    Part: [description of the specific part, or "N/A" if describing a whole object]

Analysis:""",
)

default_conversation = conv_vicuna_v0
conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llama_2": conv_llama_2,
    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "llava_v0": conv_llava_v0,
    "v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,
    "mpt": conv_mpt,
    "chatml": conv_chatml,
    "part_whole": conv_llava_part_whole,  # New template
}


# Example usage function for part-whole segmentation
def process_segmentation_task(user_input: str, image_path: str, llava_model, sam_model):
    """
    Example function showing how to use the part-whole segmentation functionality.
    """
    # Initialize conversation with part-whole capability
    conv = conv_templates["part_whole"].copy()
    
    # Process the user input to determine if it's part-whole
    whole_desc, part_desc, is_part_whole = conv.process_segmentation_input(
        user_input, llava_model
    )
    
    if is_part_whole:
        print(f"Part-whole segmentation detected:")
        print(f"Whole: {whole_desc}")
        print(f"Part: {part_desc}")
        
        # Generate two sets of seg_tokens - one for whole, one for part
        # This is pseudocode - implement according to your LLaVA architecture
        whole_tokens = generate_seg_tokens(whole_desc, image_path, llava_model)
        part_tokens = generate_seg_tokens(part_desc, image_path, llava_model)
        
        # Generate masks using SAM
        whole_mask = sam_model.decode(whole_tokens)
        part_mask = sam_model.decode(part_tokens)
        
        # Compute intersection for final mask
        final_mask = compute_mask_intersection(whole_mask, part_mask)
        
        return final_mask, True
    else:
        print(f"Whole object segmentation: {whole_desc}")
        
        # Standard processing for whole objects
        seg_tokens = generate_seg_tokens(whole_desc, image_path, llava_model)
        final_mask = sam_model.decode(seg_tokens)
        
        return final_mask, False


def generate_seg_tokens(description: str, image_path: str, llava_model):
    """Placeholder function for generating segmentation tokens."""
    # Implement according to your LLaVA-SAM architecture
    pass


def compute_mask_intersection(mask1, mask2):
    """Compute intersection of two masks."""
    # Implement mask intersection logic
    # return mask1 & mask2  # Example for binary masks
    pass


if __name__ == "__main__":
    print(default_conversation.get_prompt())
    
    # Test part-whole functionality
    conv = conv_templates["part_whole"].copy()
    test_input = "person's head"
    
    # This would require actual LLaVA model for testing
    # whole, part, is_part = conv.
    # (test_input, None)
    # print(f"Input: {test_input}")
    # print(f"Whole: {whole}, Part: {part}, Is part-whole: {is_part}")