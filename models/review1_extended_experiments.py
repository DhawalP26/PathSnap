"""Run the extended Review 1 ablation with the original benchmark split."""
from models import review1_experiments as runner
from models.review1_extended_features import EXTENDED_VARIANTS, extend

def main():
    variants = {"full_context": runner.VARIANTS["full_context"], **EXTENDED_VARIANTS}

    runner.main(variants=variants, feature_builder=extend)

if __name__ == "__main__":
    main()
