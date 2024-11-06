from typing import Dict, List
from translation.translation import TranslationPhase
from selection.selection import SelectionPhase
# from post_processing.post_processing import PostProcessingPhase
import argparse
from dotenv import load_dotenv
load_dotenv()


class PipelineManager:
    def __init__(
        self, translation_args: Dict[str, List[str]], selection_args: Dict[str, List[str]], post_processing_args: Dict[str, str]
    ):
        self.translation_phase = TranslationPhase(**translation_args)
        self.selection_phase = SelectionPhase(**selection_args)
        # self.post_processing_phase = PostProcessingPhase(
        #     **post_processing_args)

    def run(self, only_translation=False, only_selection=False, only_post_processing=False):
        if only_translation:
            self.translation_phase.run()
        elif only_selection:
            self.selection_phase.run()
            pass
        elif only_post_processing:
            # self.post_processing_phase.run()
            pass
        else:
            self.translation_phase.run()
            self.selection_phase.run()
            # self.post_processing_phase.run()


def main():
    parser = argparse.ArgumentParser(
        description="Run the entire automated pipeline or specific phases.")
    parser.add_argument("--input_files", nargs="+", required=True, type=str,
                        help="Paths to the dataset files to be translated.")
    parser.add_argument("--evaluators", nargs="+", default=["llama", "gemma", "phi", "qwen", "gpt"], type=str,
                        help="LLMs to use for scoring the translations.")
    parser.add_argument("--translators", nargs="+", default=[
                        "ggtrans", "gemini", "vinai", "gpt"], type=str, help="List of translation sources to use.")

    parser.add_argument("--translations_dir", default="../../data/translation", type=str,
                        help="Directory to save files in the translation phase.")
    parser.add_argument("--selection_dir", default="../../data/selection", type=str,
                        help="Directory to save files in the selection phase.")
    parser.add_argument("--output_dir", default="../../data/final", type=str,
                        help="Directory to save final output files.")
    parser.add_argument("--model_name", default="dslim/bert-base-NER-uncased", type=str,
                        help="Model name for post-processing.")

    parser.add_argument("--only_translation", action="store_true",
                        help="Run only the translation phase.")
    parser.add_argument("--only_selection", action="store_true",
                        help="Run only the selection phase.")
    parser.add_argument("--only_post_processing", action="store_true",
                        help="Run only the post-processing phase.")

    args = parser.parse_args()

    translation_args: Dict[str, List[str]] = {
        "dataset_files": args.input_files,
        "translations_dir": args.translations_dir,
        "translators": args.translators,
    }

    selection_args: Dict[str, List[str]] = {
        "evaluators": args.evaluators,
        "translators": args.translators,
        "translations_dir": args.translations_dir,
        "selection_dir": args.selection_dir,
    }

    post_processing_args: Dict[str, str] = {
        "selection_dir": args.selection_dir,
        "output_dir": args.output_dir,
        "model_name": args.model_name,
    }

    pipeline_manager = PipelineManager(
        translation_args, selection_args, post_processing_args)
    pipeline_manager.run(args.only_translation,
                         args.only_selection, args.only_post_processing)


if __name__ == "__main__":
    main()
