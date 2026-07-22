if __name__ == '__main__':
    from huggingface_hub import try_to_load_from_cache
    from llm_config import ModelConfig

    print(try_to_load_from_cache(ModelConfig.REMOTE_MODEL_NAME, "config.json"))
    # exit()