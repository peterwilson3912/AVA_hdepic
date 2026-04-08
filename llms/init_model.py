model_zoo = {
    "qwenlm":   ("llms.QwenLM",   "QwenLM"),
    "qwenvl":   ("llms.QwenVL",   "QwenVL"),
    "gemini":   ("llms.Gemini",   "Gemini"),
    "gemma4lm": ("llms.Gemma4LM", "Gemma4LM"),
    "gemma4vl": ("llms.Gemma4VL", "Gemma4VL"),
}

def init_model(model_name, num_gpus=1):
    if model_name not in model_zoo:
        supported_models = ", ".join(model_zoo.keys())
        raise ValueError(f"Model {model_name} not found in model_zoo. Supported models: {supported_models}")
    module_path, class_name = model_zoo[model_name]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(tp=num_gpus)