package pck;

import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.meta.Prediction;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

/**
 * This example is a version of the basic CSV example, but adds the following:此示例是“基本CSV示例”的一个版本，但添加了以下内容：
 * (a) Meta data tracking - i.e., where data for each example comes from 元数据跟踪 - 即，每个示例的数据来自何处
 * (b) Additional evaluation information - getting metadata for prediction errors 其他评估信息 - 获取预测错误的元数据
 *
 * @author Alex Black
 */
public class CSVExampleEvaluationMetaData {

    public static void main(String[] args) throws  Exception {
        //First: get the dataset using the record reader. This is as per CSV example - see that example for details
        RecordReader recordReader = new CSVRecordReader(0, ',');
        recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        int labelIndex = 4;
        int numClasses = 3;
        int batchSize = 150;

        RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
        iterator.setCollectMetaData(true);  //Instruct the iterator to collect metadata, and store it in the DataSet objects
        DataSet allData = iterator.next();
        allData.shuffle(123);
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);  //Use 65% of data for training

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        //Let's view the example metadata in the training and test sets: 让我们来看一下训练和测试集中的示例元数据：
        List<RecordMetaData> trainMetaData = trainingData.getExampleMetaData(RecordMetaData.class);
        List<RecordMetaData> testMetaData = testData.getExampleMetaData(RecordMetaData.class);

        //Let's show specifically which examples are in the training and test sets, using the collected metadata
        System.out.println("  +++++ Training Set Examples MetaData +++++");
        String format = "%-20s\t%s";
        for(RecordMetaData recordMetaData : trainMetaData){
            System.out.println(String.format(format, recordMetaData.getLocation(), recordMetaData.getURI()));
            //Also available: recordMetaData.getReaderClass()
        }
        System.out.println("\n\n  +++++ Test Set Examples MetaData +++++");
        for(RecordMetaData recordMetaData : testMetaData){
            System.out.println(recordMetaData.getLocation());
        }


        //Normalize data as per basic CSV example
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData);     //Apply normalization to the training data
        normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set


        //Configure a simple model. We're not using an optimal configuration here, in order to show evaluation/errors, later
        final int numInputs = 4;
        int outputNum = 3;
        long seed = 6;

        System.out.println("模型创建中....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            .updater(new Sgd(0.1))
            .l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3).build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX).nIn(3).nOut(outputNum).build())
            .backprop(true).pretrain(false)
            .build();

        //Fit the model 配置模型
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        for( int i=0; i<50; i++ ) {
            model.fit(trainingData);
        }

        //Evaluate the model on the test set 评估测试集上的模型
        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output, testMetaData);          //Note we are passing in the test set metadata here 注意我们在这里传递 测试集 元数据
        System.out.println(eval.stats());

        //Get a list of prediction errors, from the Evaluation object 从Evaluation对象中获取预测错误列表
        //Prediction errors like this are only available after calling iterator.setCollectMetaData(true) 像这样的预测错误仅在调用iterator.setCollectMetaData（true）后才可用
        List<Prediction> predictionErrors = eval.getPredictionErrors();
        System.out.println("\n\n+++++ Prediction Errors +++++");
        for(Prediction p : predictionErrors){
            System.out.println("Predicted class: " + p.getPredictedClass() + ", Actual class: " + p.getActualClass()
                + "\t" + p.getRecordMetaData(RecordMetaData.class).getLocation());
        }

        //We can also load a subset of the data, to a DataSet object: 我们还可以将数据的子集加载到DataSet对象：
        List<RecordMetaData> predictionErrorMetaData = new ArrayList<>();
        for( Prediction p : predictionErrors ) predictionErrorMetaData.add(p.getRecordMetaData(RecordMetaData.class));
        DataSet predictionErrorExamples = iterator.loadFromMetaData(predictionErrorMetaData);
        normalizer.transform(predictionErrorExamples);  //Apply normalization to this subset 将规范化应用于此子集

        //We can also load the raw data: 我们还可以加载原始数据：
        List<Record> predictionErrorRawData = recordReader.loadFromMetaData(predictionErrorMetaData);

        //Print out the prediction errors, along with the raw data, normalized data, labels and network predictions: 打印出预测错误，以及原始数据，规范化数据，标签和网络预测：
        for(int i=0; i<predictionErrors.size(); i++ ){
            Prediction p = predictionErrors.get(i);
            RecordMetaData meta = p.getRecordMetaData(RecordMetaData.class);
            INDArray features = predictionErrorExamples.getFeatures().getRow(i);
            INDArray labels = predictionErrorExamples.getLabels().getRow(i);
            List<Writable> rawData = predictionErrorRawData.get(i).getRecord();

            INDArray networkPrediction = model.output(features);

            System.out.println(meta.getLocation() + ": "
                + "\tRaw Data: " + rawData
                + "\tNormalized: " + features
                + "\tLabels: " + labels
                + "\tPredictions: " + networkPrediction);
        }


        //Some other useful evaluation methods: 其他一些有用的评估方法：
        List<Prediction> list1 = eval.getPredictions(1,2);      //Predictions: actual class 1, predicted class 2
        List<Prediction> list2 = eval.getPredictionByPredictedClass(2);     //All predictions for predicted class 2
        List<Prediction> list3 = eval.getPredictionsByActualClass(2);       //All predictions for actual class 2
    }
}
