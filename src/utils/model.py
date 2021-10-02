import logging

from transformers import GPT2LMHeadModel
from transformers import GPT2Config
from transformers import GPT2Tokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_student_state_dict(
    teacher_state_dict,
    n_teacher_layers,
    prefix: str = "transformer",
    use_idx: list = None,
    start_idx: int = 0
):
    student_state_dict = {}

    if not use_idx:
        if start_idx == 0:
            mid = n_teacher_layers // 2
            use_idx = list(range(start_idx, mid, 2)) + \
                list(range(mid+1, n_teacher_layers, 2))
        elif start_idx == 1:
            use_idx = list(range(start_idx, n_teacher_layers, 2))

    for param_name in ["wte.weight", "wpe.weight"]:
        student_state_dict[f"{prefix}.{param_name}"] = teacher_state_dict[f"{prefix}.{param_name}"]

    for w in ["weight", "bias"]:
        student_state_dict[f"{prefix}.ln_f.{w}"] = teacher_state_dict[f"{prefix}.ln_f.{w}"]
    
    student_state_dict["lm_head.weight"] = teacher_state_dict["lm_head.weight"]
    
    std_idx = 0
    for teacher_idx in use_idx:
        for layer in ["ln_1", "attn.c_attn", "attn.c_proj", "ln_2", "mlp.c_fc", "mlp.c_proj"]:
            for w in ["weight", "bias"]:
                student_state_dict[f"{prefix}.h.{std_idx}.{layer}.{w}"] = teacher_state_dict[
                    f"{prefix}.h.{teacher_idx}.{layer}.{w}"
                ]
            student_state_dict[f"{prefix}.h.{std_idx}.attn.bias"] = teacher_state_dict[
                f"{prefix}.h.{teacher_idx}.attn.bias"
            ]
        std_idx += 1
    
    logger.info(f"N layers selected for distillation: {std_idx}")
    logger.info(f"Number of params transferred for distillation: {len(student_state_dict.keys())}")
    return student_state_dict


def get_models(
    from_hf: bool =False,
    get_student: bool = False,
    model_name: str = None,
    teacher_config_json: str = None,
    teacher_pretrained_weights: str = None,
    student_config_json: str = None
):
    """Initialize and return teacher and student models"""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    student = None
    
    if from_hf:
        if not model_name:
            raise ValueError(
                "`model-name` is required to load model from HuggingFace")
        teacher = GPT2LMHeadModel.from_pretrained(model_name)
    
    else:
        teacher_config = GPT2Config.from_pretrained(teacher_config_json)
        teacher = GPT2LMHeadModel.from_pretrained(teacher_pretrained_weights, config=teacher_config)

    teacher.config.output_hidden_states = True
    
    if student_config_json or get_student:
        student_model_state_dict = get_student_state_dict(
            teacher.state_dict(),
            teacher.config.n_layer,
            prefix="transformer",
            start_idx=1
        )
    
    if student_config_json or get_student:
        student_config = GPT2Config.from_pretrained(student_config_json)
        student_config.output_hidden_states = True        
        student = GPT2LMHeadModel(student_config)
        
    if student:
        # TODO: Check necessity
        assert student.config.vocab_size == teacher.config.vocab_size
        assert student.config.hidden_size == teacher.config.hidden_size
        assert student.config.max_position_embeddings == teacher.config.max_position_embeddings
        
        student.load_state_dict(student_model_state_dict, strict=False)
        student.config.output_hidden_states = True
    
    return tokenizer, teacher, student
