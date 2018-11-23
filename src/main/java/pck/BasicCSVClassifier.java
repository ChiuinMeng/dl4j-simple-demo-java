package pck;

import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 该示例旨在是一种简单的CSV分类器，其将来自用于动物分类的测试数据的训练数据分开。
 * 它适合作为初学者的例子，因为它不仅将CSV数据加载到网络中，
 * 还展示了如何提取数据和显示分类结果，
 * 以及从测试数据中映射标签的简单方法 进入结果。
 */
public class BasicCSVClassifier {

    private static Logger log = LoggerFactory.getLogger(BasicCSVClassifier.class);

    private static Map<Integer,String> eats = readEnumCSV("eats.csv");
    private static Map<Integer,String> sounds = readEnumCSV("sounds.csv");
    private static Map<Integer,String> classifiers = readEnumCSV("classifiers.csv");

    public static void main(String[] args){

        try {
            // 第二：RecordReaderDataSetIterator处理转换为DataSet对象，准备在神经网络中使用
            int labelIndex = 4;      // animals.csv CSV每行中有5个值：4个输入要素，后跟整数标签（类）索引。 标签是每行中的第5个值（索引4）
            int numClasses = 3;      // 动物数据集中的3类（动物类型）。 类具有整数值0,1或2


            int batchSizeTraining = 30;    // 鸢尾数据集：总共150个例子。 我们将所有这些加载到一个DataSet中（不推荐用于大型数据集）
            DataSet trainingData = readCSVDataset(
                    "animals_train.csv",
                    batchSizeTraining, labelIndex, numClasses);

            // 这是我们想要分类的数据
            int batchSizeTest = 44;
            DataSet testData = readCSVDataset("animals.csv",
                    batchSizeTest, labelIndex, numClasses);

            // 在规范化之前为记录创建数据模型，因为它会更改数据。
            Map<Integer,Map<String,Object>> animals = makeAnimalsForTesting(testData);


            // 我们需要规范化我们的数据。 我们将使用NormalizeStandardize（它给出我们的平均值0，单位方差）
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(trainingData);           //从训练数据中收集统计数据（mean / stdev）。 这不会修改输入数据
            normalizer.transform(trainingData);     //将标准化应用于训练数据
            normalizer.transform(testData);         //将标准化应用于测试数据。 这是使用从* training * set计算的统计数据
            // 对许多机器学习模型而言，我们都必须先将数据标准化，确保模型不会因离群值而发生扭曲。
            // 数据标准化指将不同数量级的值（变化幅度为几十、几百或几百万）缩放至一个常见的范围内，比如0到1之间。
            // 只有在同一个范围内才能进行同类之间的比较……

            final int numInputs = 4;
            int outputNum = 3;
            int epochs = 1000;
            long seed = 6;

            log.info("模型创建中....");
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .activation(Activation.TANH)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Sgd(0.1))
                    .l2(1e-4)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3).build())
                    .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .activation(Activation.SOFTMAX).nIn(3).nOut(outputNum).build())
                    .backprop(true).pretrain(false)
                    .build();

            // 跑模型
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setListeners(new ScoreIterationListener(100));

            for( int i=0; i<epochs; i++ ) {
                model.fit(trainingData);
            }

            // 评估测试集上的模型
            // DL4J用一个Evaluation对象来收集有关模型性能表现的统计信息。
            // 网络定型完毕后，您就会看到这样的F1值。在机器学习中，F1值是用于衡量分类器性能的指标之一。
            // F1值是一个零到一之间的数值，可以说明网络在定型过程中表现如何。
            // 它与百分比相类似，F1值为1相当于100%的预测准确率。F1值基本上相当于神经网络作出准确预测的概率。
            Evaluation eval = new Evaluation(3);
            INDArray output = model.output(testData.getFeatures());

            eval.eval(testData.getLabels(), output);
            log.info(eval.stats());

            setFittedClassifiers(output, animals);
            logAnimals(animals);

        } catch (Exception e){
            e.printStackTrace();
        }
    }



    public static void logAnimals(Map<Integer,Map<String,Object>> animals){
        for(Map<String,Object> a:animals.values())
            log.info(a.toString());
    }

    public static void setFittedClassifiers(INDArray output, Map<Integer,Map<String,Object>> animals){
        for (int i = 0; i < output.rows() ; i++) {

            // 从拟合结果中设置分类
            animals.get(i).put("classifier",
                    classifiers.get(maxIndex(getFloatArrayFromSlice(output.slice(i)))));

        }

    }

    /**
     * 此方法用于说明如何将INDArray转换为float数组。
     * 这是为了提供更多关于如何将INDArray转换为更加以Java为中心的类型的示例。
     */
    public static float[] getFloatArrayFromSlice(INDArray rowSlice){
        float[] result = new float[rowSlice.columns()];
        for (int i = 0; i < rowSlice.columns(); i++) {
            result[i] = rowSlice.getFloat(i);
        }
        return result;
    }

    /**
     * 找到最大项目索引。
     * 这在数据拟合时使用，我们想确定将测试行分配给哪个类
     */
    public static int maxIndex(float[] vals){
        int maxIndex = 0;
        for (int i = 1; i < vals.length; i++){
            float newnumber = vals[i];
            if ((newnumber > vals[maxIndex])){
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * 获取为matric加载的数据集并从中创建记录模型，以便我们可以将拟合的分类器与记录相关联。
     */
    public static Map<Integer,Map<String,Object>> makeAnimalsForTesting(DataSet testData){
        Map<Integer,Map<String,Object>> animals = new HashMap<>();

        INDArray features = testData.getFeatures();
        for (int i = 0; i < features.rows() ; i++) {
            INDArray slice = features.slice(i);
            Map<String,Object> animal = new HashMap();

            // 设置属性
            animal.put("yearsLived", slice.getInt(0));
            animal.put("eats", eats.get(slice.getInt(1)));
            animal.put("sounds", sounds.get(slice.getInt(2)));
            animal.put("weight", slice.getFloat(3));

            animals.put(i,animal);
        }
        return animals;

    }


    public static Map<Integer,String> readEnumCSV(String csvFileClasspath) {
        try{
            List<String> lines = IOUtils.readLines(new ClassPathResource(csvFileClasspath).getInputStream());
            Map<Integer,String> enums = new HashMap<>();
            for(String line:lines){
                String[] parts = line.split(",");
                enums.put(Integer.parseInt(parts[0]),parts[1]);
            }
            return enums;
        } catch (Exception e){
            e.printStackTrace();
            return null;
        }
    }

    /**
     * 用于测试和训练
     */
    private static DataSet readCSVDataset(
            String csvFileClasspath, int batchSize, int labelIndex, int numClasses)
            throws IOException, InterruptedException{

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource(csvFileClasspath).getFile()));
        DataSetIterator iterator = new RecordReaderDataSetIterator(rr,batchSize,labelIndex,numClasses);
        return iterator.next();
    }
}
