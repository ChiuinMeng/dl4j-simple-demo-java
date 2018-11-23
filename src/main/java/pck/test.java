package pck;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.nd4j.linalg.ops.transforms.Transforms.abs;
import static org.nd4j.linalg.ops.transforms.Transforms.exp;

/**
 * 多类逻辑回归。
 * 要成功应用此算法，类必须是“线性可分”的。
 * 与Naive Bayes不同，它不承担强大的功能独立性。
 * 此示例可用于介绍机器学习。
 * 神经网络可以看作是该模型的非线性扩展。
 */
public class test {

    private static String fileName = "鸢尾数据.txt";  //可以换成wine.data
    private static int classStartPos = 5;  //wine.data是0；鸢尾数据.txt是5。即一行中判断类别的数据是第几个（0开始）
    private static int dataSize = 150;  //wine.data是178；鸢尾数据.txt是150。

    private static final Logger log = LoggerFactory.getLogger(test.class);

    // 程序的入口
    public static void main(String[] args) {
        try{
            if(args.length!=0){
                fileName = args[0];
                classStartPos = Integer.valueOf(args[1]);
                dataSize = Integer.valueOf(args[2]);
            }
        }catch (Exception e){
            e.printStackTrace();
            System.out.println("参数输入有误，重新输入！重置为默认参数\n");
        }

        // 获取研究的数据集，从文件中获取。80%用于训练，20%用于测试
        int t1 = (int)Math.round(dataSize*0.8);
        int t2 = dataSize-t1;
        DataSet examinationDataSet = getExaminationDataSet(dataSize);
        SplitTestAndTrain trainAndTest = examinationDataSet.splitTestAndTrain(t1, new Random(dataSize-t1));
        DataSet trainingData = trainAndTest.getTrain();
        DataSet testData = trainAndTest.getTest();
        System.out.println("获取数据完成，数据项数为："+dataSize+",训练个数："+t1+"，测试个数："+t2);
        // 一些参数
        long maxIterations = 10000; //最大迭代次数
        double learningRate = 0.01;  //学习速率
        double minLearningRate = 0.0001;  //最低学习速率
        // 训练模型，然后测试模型
        INDArray model = trainModel(trainingData, maxIterations, learningRate, minLearningRate);
        testModel(testData, model);
    }


    // 从文件中获取研究数据集
    // DL4J采用一种称为DataSet的对象来将数据加载到神经网络中。
    // DataSet可以存储需要进行预测的数据（及其相关标签）一个DataSet对象中包含两个NDArray，NDArray即N维数组，是DL4J用于数值计算的基本对象。
    private static DataSet getExaminationDataSet(int dataSize) {
        DataSet examinationDataSet = null;
        try {
            File examinationData = new ClassPathResource(fileName).getFile();
            BufferedReader reader = new BufferedReader(new FileReader(examinationData));
            List<DataSet> data = reader.lines()
                    .filter(l -> !l.trim().isEmpty())
                    .map(mapRowToDataSet)
                    .collect(Collectors.toList());
            if (reader != null)
                reader.close();
            DataSetIterator iter = new IteratorDataSetIterator(data.iterator(), dataSize);
            examinationDataSet = iter.next();
        } catch (IOException e) {
            log.error("IO异常！", e);
        }
        return examinationDataSet;
    }

    /* 请注意，应用程序可以将datavec用于此类转换，尤其是对于大数据集. */
    private static Function<String, DataSet> mapRowToDataSet = (String line) -> {
        //sepalLengthCm,sepalWidthCm,petalLengthCm,petalWidthCm,species
        double[] parsedRows = Arrays.stream(line.split(","))
                .mapToDouble(v -> {
                    switch (v) {
                        case "Iris-setosa":    //视文件而定，这里写死了。但不影响wine.data，因为那全是数据。
                            return 0.0;
                        case "Iris-versicolor":
                            return 1.0;
                        case "Iris-virginica":
                            return 2.0;
                        default:
                            return Double.parseDouble(v);
                    }
                }).toArray();
        // 上面过程后得到的parsedRows是一维double型数组
        int columns = parsedRows.length;
        double[] temp = new double[columns-1];
        System.arraycopy(parsedRows,0,temp,0,classStartPos-0);
        System.arraycopy(parsedRows,classStartPos+1,temp,classStartPos,columns-classStartPos-1);
        return new DataSet(  // 红酒和鸢尾在这里会有不同！红酒数组第一项为类别，鸢尾最后一项为类别。所以需要参数classStartPos。
                Nd4j.create(temp),  //输入参数
                Nd4j.create( Arrays.copyOfRange(parsedRows,classStartPos,classStartPos+1)));  //输出类别
    };

    /**
     * 训练模型
     */
    public static INDArray trainModel(DataSet trainDataSet, long maxIterations, double learningRate,
                                      double minLearningRate) {
        log.info("训练模型中...");
        long start = System.currentTimeMillis();
        INDArray trainFeatures = prependConstant(trainDataSet);
        INDArray trainLabels = trainDataSet.getLabels();

        // 为了处理多个类，我们为每个类构建一个模型，可以预测一个例子是否属于它，然后我们将选择具有最高概率的类来给出最终的类预测
        int labelSize = 3; //红酒和鸢尾都是三个类别。
        INDArray classLabels[] = new INDArray[labelSize];
        INDArray models[] = new INDArray[labelSize];
        for(int t=0; t<labelSize; t++){
            classLabels[t] = getClassLabels(trainLabels, t);
            // 训练该模型。
            models[t] = training(trainFeatures, classLabels[t], maxIterations, learningRate, minLearningRate);
        }

        INDArray finalModel = Nd4j.hstack(models);
        log.debug("模型的参数s:\n{}", finalModel);
        log.info("花费时间 {} ms", (System.currentTimeMillis() - start));
        return finalModel;
    }


    /**
     *  测试模型
     * @param testDataSet 测试集
     * @param params 参数
     */
    public static void testModel(DataSet testDataSet, INDArray params) {
        log.info("测试模型中...");
        INDArray testFeatures = prependConstant(testDataSet);
        INDArray testLabels = testDataSet.getLabels();
        INDArray predictedLabels = predictLabels(testFeatures, params);

        long numOfSamples = testLabels.size(0);
        double correctSamples = countCorrectPred(testLabels, predictedLabels);
        double accuracy = correctSamples / numOfSamples;
        log.info("正确的样品: {}/{}", (int) correctSamples, numOfSamples);
        log.info("精准度: {}", accuracy);
    }

    /**
     * 前置线性回归的常数项（一列）。
     * 这避免了所有特征都为零的情况，产生零预测，这意味着50％的概率（即最大不确定性）。
     * @param dataset 数据集
     * @return 特征
     */
    public static INDArray prependConstant(DataSet dataset) {
        INDArray features = Nd4j.hstack(
                Nd4j.ones(dataset.getFeatures().size(0), 1),
                dataset.getFeatures());
        return features;
    }

    /**
     * Logistic函数
     * 计算每个元素的0到1之间的数字。
     * 注意ND4J带有自己的sigmoid函数。
     * @param y 输入值
     * @return 概率
     */
    private static INDArray sigmoid(INDArray y) {
        y = y.mul(-1.0);
        y = exp(y, false);
        y = y.add(1.0);
        y = y.rdiv(1.0);
        return y;
    }

    /**
     * 二元逻辑回归。
     * 计算一个例子是某种花的概率。
     * 可以一次计算一批示例，即将样本作为行和列作为特征的矩阵（这通常由DL4J内部完成）。
     * @param x 特征
     * @param p 参数
     * @return 类别概率
     */
    private static INDArray predict(INDArray x, INDArray p) {
        INDArray y = x.mmul(p); //线性回归
        return sigmoid(y);
    }

    /**
     * Gradient函数
     * 计算成本函数的梯度以及每个参数在结果中输入的误差。
     * @param x 特征
     * @param y 标签
     * @param p 参数
     * @return 参数梯度
     */
    private static INDArray gradient(INDArray x, INDArray y, INDArray p) {
        long m = x.size(0); //x数组大小，样例空间大小。
        INDArray pred = predict(x, p);
        INDArray diff = pred.dup().sub(y); //预测类别和实际类别的不同
        return x.dup()
                .transpose()
                .mmul(diff)
                .mul(1.0 / m);
    }

    /**
     * 训练算法
     * 梯度下降优化。
     * 逻辑成本函数（或最大熵）是凸的，因此我们保证找到全局最小值。
     *
     * @param x 输入特征
     * @param y 输出类别
     * @param maxIterations 最大学习迭代次数
     * @param learningRate 参数变化有多快
     * @param minLearningRate 学习率下限
     * @return 最佳参数
     */
    private static INDArray training(INDArray x, INDArray y, long maxIterations, double learningRate,
                                     double minLearningRate) {
        Nd4j.getRandom().setSeed(12345);
        INDArray params = Nd4j.rand((int)x.size(1), 1); //随机猜的

        INDArray newParams = params.dup();
        INDArray optimalParams = params.dup();

        for (int i = 0; i < maxIterations; i++) {
            INDArray gradients = gradient(x, y, params);
            gradients = gradients.mul(learningRate);
            newParams = params.sub(gradients);

            if (hasConverged(params, newParams, minLearningRate)) {
                log.debug("完成迭代次数: {}", i + 1);
                break;
            }
            params = newParams;
        }

        optimalParams = newParams;
        return optimalParams;
    }

    // 判断是否融合了，拟合了。用于判断迭代次数。
    private static boolean hasConverged(INDArray oldParams, INDArray newParams, double threshold) {
        double diffSum = abs(oldParams.sub(newParams)).sumNumber().doubleValue();
        return diffSum / oldParams.size(0) < threshold;
    }

    private static INDArray getClassLabels(INDArray labels, double label) {
        INDArray binaryLabels = labels.dup();
        for (int i = 0; i < binaryLabels.rows(); i++) {
            double v = binaryLabels.getDouble(i);
            if (v == label)
                binaryLabels.putScalar(i, 1.0);
            else
                binaryLabels.putScalar(i, 0.0);
        }
        return binaryLabels;
    }

    /**
     * 标签预测
     * 最大后验概率估计。
     * 对于每个示例：运行N个独立预测（每个类一个）并返回具有最高值（argmax）的预测。
     */
    private static INDArray predictLabels(INDArray features, INDArray params) {
        INDArray predictions = features.mmul(params).argMax(1);
        return predictions;
    }

    private static double countCorrectPred(INDArray labels, INDArray predictions) {
        double counter = 0;
        for (int i = 0; i < labels.size(0); i++) {
            if (labels.getDouble(new int[] { i }) == predictions.getDouble(new int[] { i })) {
                counter++;
            }
        }
        return counter;
    }

}
