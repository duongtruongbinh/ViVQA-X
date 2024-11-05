from typing import Dict
from translation.translation import TranslationPhase
# from .selection.selection import SelectionPhase
# from .post_processing.post_processing import PostProcessingPhase
import argparse
from dotenv import load_dotenv
load_dotenv()


class PipelineManager:
    def __init__(
        self, translation_args: Dict, selection_args: Dict, post_processing_args: Dict
    ):
        self.translation_phase = TranslationPhase(**translation_args)
        # self.selection_phase = SelectionPhase(**selection_args)
        # self.post_processing_phase = PostProcessingPhase(
        #     **post_processing_args)

    def run(self):
        self.translation_phase.run()
        # self.selection_phase.run()
        # self.post_processing_phase.run()


def main():
    parser = argparse.ArgumentParser(
        description="Run the entire automated pipeline.")
    parser.add_argument("--input_files", nargs="+", required=True,
                        help="Paths to the dataset files to be translated.")
    parser.add_argument("--evaluators", nargs="+", default=["llama", "gemma", "phi", "qwen", "gpt"],
                        help="LLMs to use for scoring the translations.")
    parser.add_argument("--translators", nargs="+", default=[
                        "ggtrans", "gemini", "vinai", "gpt"], help="List of translation sources to use.")

    parser.add_argument("--translations_dir", default="../../data/translation",
                        help="Directory to save files in the translation phase.")
    parser.add_argument("--selection_dir", default="../../data/selection",
                        help="Directory to save files in the selection phase.")
    parser.add_argument("--output_dir", default="../../data/final",
                        help="Directory to save final output files.")
    parser.add_argument("--model_name", default="dslim/bert-base-NER-uncased",
                        help="Model name for post-processing.")

    args = parser.parse_args()

    translation_args = {
        "dataset_files": args.input_files,
        "translations_dir": args.translations_dir,
        "translators": args.translators,
    }

    selection_args = {
        "evaluators": args.evaluators,
        "translations_dir": args.translations_dir,
        "selection_dir": args.selection_dir,
    }

    post_processing_args = {
        "selection_dir": args.selection_dir,
        "output_dir": args.output_dir,
        "model_name": args.model_name,
    }

    pipeline_manager = PipelineManager(
        translation_args, selection_args, post_processing_args)
    pipeline_manager.run()


if __name__ == "__main__":
    main()
