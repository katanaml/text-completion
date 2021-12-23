const { BertWordPieceTokenizer } = require("tokenizers");

async function main() {
    const wordPieceTokenizer = await BertWordPieceTokenizer.fromOptions({ vocabFile: 'vocab.txt' });
    const wpEncoded = await wordPieceTokenizer.encode("Paris is the [MASK] of France.");

    console.log(wpEncoded.length);
    console.log(wpEncoded.tokens);
    console.log(wpEncoded.ids);
    console.log(wpEncoded.attentionMask);
    console.log(wpEncoded.offsets);
    console.log(wpEncoded.overflowing);
    console.log(wpEncoded.specialTokensMask);
    console.log(wpEncoded.typeIds);
    console.log(wpEncoded.wordIndexes);
}

main();