from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor

class ImageCaptionerModel:
    def __init__(self, encoder_model: str, decoder_model: str):
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model, decoder_model)
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_model)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def get_model_components(self):
        return self.model, self.tokenizer, self.feature_extractor
