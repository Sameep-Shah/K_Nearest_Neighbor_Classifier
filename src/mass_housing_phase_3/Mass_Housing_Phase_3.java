package mass_housing_phase_3;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AddClassification;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author sameepshah
 */
public class Mass_Housing_Phase_3 {
    
    private void getFile(String fName) {
        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(classLoader.getResource(fName).getFile());
    }
    
    public static void main(String[] args) throws Exception {
        
        String s = args[0];
        String t = args[1];
        String u = args[2];
        String v = args[3];        
        
        Mass_Housing_Phase_3 m = new Mass_Housing_Phase_3();
        
        m.getFile(s);
        m.getFile(t);
        m.getFile(u);
        m.getFile(v);
        
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(s));
        Instances data = loader.getDataSet();
        
        data.setClassIndex(8);
        
        Remove rem;
        rem = new Remove();
        rem.setAttributeIndices("9,10,20-last");
        rem.setInputFormat(data);
        data = Filter.useFilter(data, rem);
        
        Normalize nm = new Normalize();
        nm.setInputFormat(data);
        data = Filter.useFilter(data, nm);
        

        //   ClusterMembership cm = new ClusterMembership();
        // cm.setInputFormat(data);
        // data = Filter.useFilter(data, cm);
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("Mass_Housing_Phase_3/phase3_train.arff"));
        saver.setDestination(new File("Mass_Housing_Phase_3/phase3_train.arff"));
        saver.writeBatch();
        
        CSVLoader loader1 = new CSVLoader();
        loader1.setSource(new File(t));
        Instances data1 = loader1.getDataSet();
        
        data1.setClassIndex(8);
        
        rem = new Remove();
        rem.setAttributeIndices("9,10,20-last");
        rem.setInputFormat(data1);
        data1 = Filter.useFilter(data1, rem);

        // cm.setInputFormat(data1);
        // data1 = Filter.useFilter(data, cm);
        nm.setInputFormat(data1);
        data1 = Filter.useFilter(data1, nm);
        
        ArffSaver saver1 = new ArffSaver();
        saver1.setInstances(data1);
        saver1.setFile(new File("Mass_Housing_Phase_3/phase3_test.arff"));
        saver1.setDestination(new File("Mass_Housing_Phase_3/phase3_test.arff"));
        saver1.writeBatch();
        
        CSVLoader loader2 = new CSVLoader();
        loader2.setSource(new File(u));
        Instances data2 = loader2.getDataSet();
        
        rem = new Remove();
        rem.setAttributeIndices("9,10,20-last");
        rem.setInputFormat(data2);
        data2 = Filter.useFilter(data2, rem);
        
        nm.setInputFormat(data2);
        data2 = Filter.useFilter(data2, nm);
        
        ArffSaver saver2 = new ArffSaver();
        saver2.setInstances(data2);
        saver2.setFile(new File("Mass_Housing_Phase_3/phase3_train_full.arff"));
        saver2.setDestination(new File("Mass_Housing_Phase_3/phase3_train_full.arff"));
        saver2.writeBatch();
        
        CSVLoader loader3 = new CSVLoader();
        loader3.setSource(new File(v));
        Instances data3 = loader3.getDataSet();
        
        rem = new Remove();
        rem.setAttributeIndices("9,10,20-last");
        rem.setInputFormat(data3);
        data3 = Filter.useFilter(data3, rem);
        
        nm.setInputFormat(data3);
        data3 = Filter.useFilter(data3, nm);
        
        ArffSaver saver3 = new ArffSaver();
        saver3.setInstances(data3);
        saver3.setFile(new File("Mass_Housing_Phase_3/phase3_new_test.arff"));
        saver3.setDestination(new File("Mass_Housing_Phase_3/phase3_new_test.arff"));
        saver3.writeBatch();
        
        BufferedReader datafile1 = new BufferedReader(new FileReader("Mass_Housing_Phase_3/phase3_train.arff"));
        BufferedReader datafile2 = new BufferedReader(new FileReader("Mass_Housing_Phase_3/phase3_test.arff"));
        BufferedReader datafile3 = new BufferedReader(new FileReader("Mass_Housing_Phase_3/phase3_train_full.arff"));
        BufferedReader datafile4 = new BufferedReader(new FileReader("Mass_Housing_Phase_3/phase3_new_test.arff"));
        
        Instances train = new Instances(datafile1);
        Instances test = new Instances(datafile2);
        Instances train_full = new Instances(datafile3);
        Instances test_new = new Instances(datafile4);
        
        Classifier cls1 = new IBk(7);
        Classifier cls2 = new IBk(7);
        int folds = 5;
        
        train.setClassIndex(7);
        test.setClassIndex(7);
        train_full.setClassIndex(7);
        test_new.setClassIndex(7);
        
         Instances predictedData1 = null;
            Instances predictedData2 = null;

        /*Random rand1 = new Random(Integer.MAX_VALUE);
            Instances randData1 = new Instances(train);
            randData1.randomize(rand1);
            randData1.setClassIndex(6);
            if (randData1.classAttribute().isNominal()) {
                randData1.stratify(folds);
            }
            Random rand2 = new Random(Integer.MAX_VALUE);
            Instances randData2 = new Instances(test);
            randData2.randomize(rand2);
            randData2.setClassIndex(6);
            if (randData2.classAttribute().isNominal()) {
                randData2.stratify(folds);
            }
            Random rand3 = new Random(Integer.MAX_VALUE);
            Instances randData3 = new Instances(train_full);
            randData3.randomize(rand3);
            randData3.setClassIndex(6);
            if (randData3.classAttribute().isNominal()) {
                randData3.stratify(folds);
            }
            Random rand4 = new Random(Integer.MAX_VALUE);
            Instances randData4 = new Instances(test_new);
            randData4.randomize(rand4);
            randData4.setClassIndex(6);
            if (randData4.classAttribute().isNominal()) {
                randData4.stratify(folds);
            }*/
        Evaluation eval1 = new Evaluation(train);
        Evaluation eval2 = new Evaluation(train_full);

        //  for (int n = 0; n < folds; n++) {
        // Instances train1 = train.trainCV(folds, n);
        // Instances test1 = test.testCV(folds, n);
        // Instances train2 = train_full.trainCV(folds, n);
        // Instances test2 = test_new.testCV(folds, n);
        cls1.buildClassifier(train);
        eval1.evaluateModel(cls1, test);
        cls2.buildClassifier(train_full);
        eval2.evaluateModel(cls2, test_new);
        
        AddClassification filter1 = new AddClassification();
        filter1.setClassifier(cls1);
        filter1.setOutputClassification(true);
        filter1.setOutputDistribution(true);
        filter1.setOutputErrorFlag(true);
        filter1.setInputFormat(train);
        Filter.useFilter(train, filter1);
        
        AddClassification filter2 = new AddClassification();
        filter2.setClassifier(cls2);
        filter2.setOutputClassification(true);
        filter2.setOutputDistribution(true);
        filter2.setOutputErrorFlag(true);
        filter2.setInputFormat(train_full);
        Filter.useFilter(train_full, filter2);
        
          Instances pred1 = Filter.useFilter(test, filter1);
                if (predictedData1 == null) {
                    predictedData1 = new Instances(pred1, 0);
                }
                for (int j = 0; j < pred1.numInstances(); j++) {
                    predictedData1.add(pred1.instance(j));

                }
                
                Instances pred2 = Filter.useFilter(test_new, filter2);
                if (predictedData2 == null) {
                    predictedData2 = new Instances(pred2, 0);
                }
                for (int j = 0; j < pred2.numInstances(); j++) {
                    predictedData2.add(pred2.instance(j));

                }
        
        System.out.println(eval1.toSummaryString());
        System.out.println(eval1.toMatrixString("confusion matrix"));
        
        System.out.println(eval2.toSummaryString());
        System.out.println(eval2.toMatrixString("confusion matrix"));
        
        rem.setAttributeIndices("2-6,9-17,19-last");
            rem.setInputFormat(predictedData1);
            predictedData1 = Filter.useFilter(predictedData1, rem);
            
            rem.setAttributeIndices("2-6,9-17,19-last");
            rem.setInputFormat(predictedData2);
            predictedData2 = Filter.useFilter(predictedData2, rem);
            
            ArffSaver saver4 = new ArffSaver();
        saver4.setInstances(predictedData2);
        saver4.setFile(new File("Mass_Housing_Phase_3/phase3predicted.arff"));
        saver4.setDestination(new File("Mass_Housing_Phase_3/phase3predicted.arff"));
        saver4.writeBatch();
            
            System.out.println("rm_key Stmt_date Financial Rating Letter Grade "
                    + " Predicted Grade");

            /*
        Loop for checking incorrect predictions
             */
            for (int i = 0; i < predictedData1.numInstances(); i++) {
                if (predictedData1.get(i).toString(2) == null ? predictedData1.get(i)
                        .toString(3) != null : !predictedData1.get(i).toString(2).
                        equals(predictedData1.get(i).toString(3))) {

                    System.out.println(predictedData1.get(i));
                    
                    

                }
            }
            
            System.out.println("rm_key\t Stmt_date\t Financial Rating Letter Grade "
                    + "\t Predicted Grade");

            for (int i = 0; i < predictedData2.numInstances(); i++) {
                if (predictedData2.get(i).toString(2) == null ? predictedData2.get(i)
                        .toString(3) != null : !predictedData2.get(i).toString(2).
                        equals(predictedData2.get(i).toString(3))) {

                    System.out.println(predictedData2.get(i));

                }
            }
            
            
    }
}
