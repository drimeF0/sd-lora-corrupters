import launch

# TODO: add pip dependency if need extra module only on extension

if not launch.is_installed("safetensors"):
    launch.run_pip("install safetensors", "requirements for MagicPrompt")