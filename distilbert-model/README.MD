# Converter &#129303;

## Description

Converter from Hugging Face DistilBERT Python model into TensorFlow SavedModel to be reusable in TensorFlow.js Node.js

## Instructions

1. Install libraries

```
pip install -r requirements.txt
```

2. Run converter

```
python main.py
```

## Structure

```
.
├── app
│   └── distilbert_to_savemodel.py
├── main.py
├── Dockerfile
├── README.md
├── saved_model
└── requirements.txt
```

## Author

[Katana ML](https://katanaml.io), [Andrej Baranovskij](https://github.com/abaranovskis-redsamurai)

## License

Licensed under the Apache License, Version 2.0. Copyright 2020-2021 Katana ML, Andrej Baranovskij. [Copy of the license](https://github.com/katanaml/text-completion/blob/main/LICENSE).
