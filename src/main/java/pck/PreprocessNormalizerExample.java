package pck;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 此基本示例演示了如何使用可用的预处理器
 * 此示例使用minmax缩放器，将与3.10版本及更高版本一起使用
 * 以后的版本和当前的master将与所有其他预处理器一起使用
 */
public class PreprocessNormalizerExample {

    private static Logger log = LoggerFactory.getLogger(PreprocessNormalizerExample.class);

    public static void main(String[] args) throws  Exception {


        //========= This section is to create a dataset and a dataset iterator from the iris dataset stored in csv =============
        //                               Refer to the csv example for details
        int numLinesToSkip = 0;
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        RecordReader recordReaderA = new CSVRecordReader(numLinesToSkip,delimiter);
        RecordReader recordReaderB = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        recordReaderA.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        recordReaderB.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        int labelIndex = 4;
        int numClasses = 3;
        DataSetIterator iteratorA = new RecordReaderDataSetIterator(recordReaderA,10,labelIndex,numClasses);
        DataSetIterator iteratorB = new RecordReaderDataSetIterator(recordReaderB,10,labelIndex,numClasses);
        DataSetIterator fulliterator = new RecordReaderDataSetIterator(recordReader,150,labelIndex,numClasses);
        DataSet datasetX = fulliterator.next();
        DataSet datasetY = datasetX.copy();

        // We now have datasetX, datasetY, iteratorA, iteratorB all of which have the iris dataset loaded
        // iteratorA and iteratorB have batchsize of 10. So the full dataset is 150/10 = 15 batches
        //=====================================================================================================================

        log.info("所有预处理器必须符合预期的指标才能用于转换");
        log.info("要在迭代器上调用next时进行转换，请使用'setpreprocessor'，例如最后的示例\n");
        log.info("此示例演示了使用min max normalizer进行预处理器的情况。");
        log.info("还提供标准化预处理器。");
        log.info("所有预处理器的用法都是相同的 - 适合然后转换数据集或设置为预处理器到迭代器");

        log.info("Instantiating a preprocessor...\n");
        NormalizerMinMaxScaler preProcessor = new NormalizerMinMaxScaler();
        log.info("During 'fit' the preprocessor calculates the metrics (std dev and mean for the standardizer, min and max for minmaxscaler) from the data given");
        log.info("Fit can take a dataset or a dataset iterator\n");

        //Fitting a preprocessor with a dataset
        log.info("Fitting with a dataset...............");
        preProcessor.fit(datasetX);
        log.info("Calculated metrics");
        log.info("Min: {}",preProcessor.getMin());
        log.info("Max: {}",preProcessor.getMax());

        log.info("Once fit the preprocessor can be used to transform data wrt to the metrics of the dataset it was fit to");
        log.info("Transform takes a dataset and modifies it in place");

        log.info("Transforming a dataset, printing only the first ten.....");
        preProcessor.transform(datasetX);
        log.info("\n{}\n",datasetX.getRange(0,9));

        log.info("Transformed datasets can be reverted back as well...");
        log.info("Note the reverting happens in place.");
        log.info("Reverting back the dataset, printing only the first ten.....");
        preProcessor.revert(datasetX);
        log.info("\n{}\n",datasetX.getRange(0,9));

        //Setting a preprocessor in an iterator
        log.info("Fitting a preprocessor with iteratorB......");
        NormalizerMinMaxScaler preProcessorIter = new NormalizerMinMaxScaler();
        preProcessorIter.fit(iteratorB);
        log.info("A fitted preprocessor can be set to an iterator so each time next is called the transform step happens automatically");
        log.info("Setting a preprocessor for iteratorA");
        iteratorA.setPreProcessor(preProcessorIter);
        while (iteratorA.hasNext()) {
            log.info("Calling next on iterator A that has a preprocessor on it");
            log.info("\n{}",iteratorA.next());
            log.info("Calling next on iterator B that has no preprocessor on it");
            DataSet firstBatch = iteratorB.next();
            log.info("\n{}",firstBatch);
            log.info("Note the data is different - iteratorA is preprocessed, iteratorB is not");
            log.info("Now using transform on the next datset on iteratorB");
            iteratorB.reset();
            firstBatch = iteratorB.next();
            preProcessorIter.transform(firstBatch);
            log.info("\n{}",firstBatch);
            log.info("Note that this now gives the same results");
            break;
        }

        log.info("If you are using batches and an iterator, set the preprocessor on your iterator to transform data automatically when next is called");
        log.info("Use the .transform function only if you are working with a small dataset and no iterator");

        log.info("MinMax scaler also takes a min-max range to scale to.");
        log.info("Instantiating a new preprocessor and setting it's min-max scale to {-1,1}");
        NormalizerMinMaxScaler preProcessorRange = new NormalizerMinMaxScaler(-1,1);
        log.info("Fitting to dataset");
        preProcessorRange.fit(datasetY);
        log.info("First ten before transforming");
        log.info("\n{}",datasetY.getRange(0,9));
        log.info("First ten after transforming");
        preProcessorRange.transform(datasetY);
        log.info("\n{}",datasetY.getRange(0,9));

    }
}
