import sacrebleu

class MetricsCalculator:
    @staticmethod
    def compute_bleu(predictions, references):
        return sacrebleu.corpus_bleu(predictions, [[ref] for ref in references]).score
