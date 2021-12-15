from app.distilbert_to_savedmodel import DistilbertConverter


def main():
    distilbert_converter = DistilbertConverter()
    distilbert_converter.call()


if __name__ == "__main__":
    main()
