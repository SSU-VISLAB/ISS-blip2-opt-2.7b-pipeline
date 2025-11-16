from transformers import Blip2ForConditionalGeneration
from pathlib import Path
import importlib.util
from export_q_former import export_q_former
from export_vision_encoder import export_vision_encoder

_spec = importlib.util.spec_from_file_location("opt_2_7b_decoder", Path(__file__).parent / "opt_2.7b_decoder.py")
_opt_decoder_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_opt_decoder_module)
export_opt_decoder = _opt_decoder_module.export_opt_decoder

def load_model():
    return Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
    )  # doctest: +IGNORE_RESULT

def main():
    model = load_model()
    model = model.to("cuda")
    Path("model").mkdir(exist_ok=True)
    export_vision_encoder(model)
    export_q_former(model)
    export_opt_decoder(model)

if __name__ == "__main__":
    main()