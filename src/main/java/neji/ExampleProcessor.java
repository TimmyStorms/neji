package neji;

import java.io.IOException;

import pt.ua.tm.gimli.corpus.Corpus;
import pt.ua.tm.neji.batch.FileBatchExecutor;
import pt.ua.tm.neji.context.Context;
import pt.ua.tm.neji.context.ContextProcessors;
import pt.ua.tm.neji.core.batch.Batch;
import pt.ua.tm.neji.core.corpus.InputCorpus;
import pt.ua.tm.neji.core.corpus.InputCorpus.InputFormat;
import pt.ua.tm.neji.core.corpus.OutputCorpus.OutputFormat;
import pt.ua.tm.neji.core.pipeline.DefaultPipeline;
import pt.ua.tm.neji.core.pipeline.Pipeline;
import pt.ua.tm.neji.core.processor.BaseProcessor;
import pt.ua.tm.neji.dictionary.Dictionary;
import pt.ua.tm.neji.dictionary.DictionaryHybrid;
import pt.ua.tm.neji.exception.NejiException;
import pt.ua.tm.neji.ml.MLHybrid;
import pt.ua.tm.neji.nlp.NLP;
import pt.ua.tm.neji.reader.RawReader;
import pt.ua.tm.neji.sentence.SentenceTagger;
import pt.ua.tm.neji.writer.NejiWriter;

public class ExampleProcessor extends BaseProcessor {

	public ExampleProcessor(final Context context, final InputCorpus inputCorpus) {
		super(context, inputCorpus);
	}

	@Override
	public void run() {
		try {
			ContextProcessors cp = getContext().take();
			Corpus corpus = getInputCorpus().getCorpus();
			Pipeline p = new DefaultPipeline();
			p.add(new RawReader());
			p.add(new SentenceTagger(cp.getSentenceSplitter()));
			p.add(new NLP(corpus, cp.getParser()));
			for (Dictionary d : getContext().getDictionaries()) {
				p.add(new DictionaryHybrid(d, corpus));
			}
			for (int i = 0; i < getContext().getModels().size(); i++) {
				p.add(new MLHybrid(corpus, getContext().getModels().get(i), cp.getCRF(i), true));
			}
			p.add(new NejiWriter(corpus));
			p.run(getInputCorpus().getInStream(), getOutputCorpus().getOutStream());
			getContext().put(cp);
		} catch (InterruptedException | NejiException | IOException e) {
			e.printStackTrace();
		}
	}

	public static void main(final String[] args) throws Exception {
		final String modelsFolder = "src/main/resources/models/";
		final String dictionariesFolder = "src/main/resources/dictionary/";
		final String inputFolder = "src/main/resources/input/";
		final String outputFolder = "output/";
		Context context = new Context(modelsFolder, dictionariesFolder);
		boolean areFilesCompressed = true;
		int numThreads = 4;
		Batch batch = new FileBatchExecutor(inputFolder, InputFormat.RAW, outputFolder, OutputFormat.NEJI,
				areFilesCompressed, numThreads, true);
		Class c = ExampleProcessor.class;
		batch.run(c, context);
	}

}