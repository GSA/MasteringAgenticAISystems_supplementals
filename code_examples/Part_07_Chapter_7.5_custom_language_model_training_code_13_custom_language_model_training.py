from nemo.collections.asr.models import EncDecRNNTBPEModel

# Load pretrained ASR model
asr_model = EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-ctc-1.1b")

# Train custom language model on medical transcripts
from nemo.collections.nlp.models import NGramLanguageModel

lm = NGramLanguageModel(
    order=5,  # 5-gram model
    smoothing="kneser_ney"
)

lm.train(
    corpus="medical_transcripts_corpus.txt",  # 100K-1M words of domain text
    vocab=asr_model.vocabulary  # Match ASR model's token vocabulary
)

# Integrate custom LM into ASR decoding
asr_model.change_decoding_strategy(
    decoder_type="ctc-decoding",
    lm_path="custom_medical_lm.binary",
    alpha=0.5,  # LM weight (0-1, higher = trust LM more)
    beta=1.5    # Word bonus (reward longer words)
)

# Use model with custom LM
transcript = asr_model.transcribe(["medical_recording.wav"])