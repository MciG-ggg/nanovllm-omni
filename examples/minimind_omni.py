from nanovllm_omni import Omni, SamplingParams

if __name__ == "__main__":
    output = Omni("jingyaogong/minimind-3o").generate(["hi"], SamplingParams(max_tokens=8))[0]
    with open("audio.wav", "wb") as f:
        f.write(output.multimodal_output["audio"].wav_bytes())
