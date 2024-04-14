using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using RestaurantSentiment.ML.Base;
using Microsoft.ML.Data;
using RestaurantSentiment.ML.Objects;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML.Trainers;
using Microsoft.ML.Calibrators;
namespace RestaurantSentiment.ML
{
    public class Trainer:BaseML
    {
        public void Train(string trainingFileName)
        {
            if(!File.Exists(trainingFileName))//Check if training data exists
            {
                Console.WriteLine($"Failed to find training data file({trainingFileName})");
                return;
            }
            //MLContext mLContext = new MLContext();//Since object reference is required for to access Data Field
            IDataView trainingDataView=mLContext.Data.LoadFromTextFile<RestaurantFeedback>(trainingFileName);
            //Loads Text file into an IDataViewObject
            DataOperationsCatalog.TrainTestData dataSplit = mLContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.2);
            //Creates Test Set from main Training Data
            //The parameter testFraction specifies the percentage of the dataset to hold back for testing in our case by 20%

         /*Creating pipeline*/

            TextFeaturizingEstimator dataProcessPipeline = mLContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(RestaurantFeedback.Text));

            //Instantiate out Trainer class
            SdcaLogisticRegressionBinaryTrainer sdcaRegressionTrainer = mLContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: nameof(RestaurantFeedback.Label), featureColumnName: "Features");

            //Complete our pipeline by appending the trainer we instantiated
            EstimatorChain<BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>>
            trainingPipeline = dataProcessPipeline.Append(sdcaRegressionTrainer);


            //Train the model with the data set created Earlier
            ITransformer trainedModel = trainingPipeline.Fit(dataSplit.TrainSet);

            //Save created model to the filename specified matching training set's schema
            mLContext.Model.Save(trainedModel, dataSplit.TrainSet.Schema, ModelPath);//Model path is supplied from BaseML


            // Transform Model with the test set created Earlier
            IDataView testSetTransform=trainedModel.Transform(dataSplit.TestSet);

            //Pass testSetTransform into BinaryClassificatioon class' Evaluate Method
            CalibratedBinaryClassificationMetrics modelMetrics = mLContext.BinaryClassification.Evaluate(
                                                                data: testSetTransform,
                                                                labelColumnName: nameof(RestaurantFeedback.Label),
                                                                scoreColumnName: nameof(FeedbackPrediction.Score));
            //Printing the main metrics using trained model with the test set
            Console.WriteLine($"Area Under Curve:{modelMetrics.AreaUnderRocCurve:P2}{Environment.NewLine}" +
              $"Area Under Precision Recall Curve: {modelMetrics.AreaUnderPrecisionRecallCurve:P2}" + $"{Environment.NewLine}" + 
              $"Accuracy:{modelMetrics.Accuracy:P2}{Environment.NewLine}" + $"F1Score:{modelMetrics.F1Score:P2}{Environment.NewLine}" + 
              $"Positive Recall:{modelMetrics.PositiveRecall:#.##}{Environment.NewLine}" + 
              $"NegativeRecall:{modelMetrics.NegativeRecall:#.##}{Environment.NewLine}");













        }
    }
}
