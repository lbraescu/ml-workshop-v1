using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace classification
{
    public class Program
    {
        private static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-data.tsv");
        private static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-test.tsv");
        private static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        private static readonly IEnumerable<SentimentData> _sentimentsToPredict = new[]
        {
            new SentimentData
            {
                SentimentText = "I never asked for this..."
            },
            new SentimentData
            {
                SentimentText = "Evozon is the best!"
            }
        };

        public static async Task Main(string[] args)
        {
            var model = await TrainModel();

            Evaluate(model);

            var predictions = model.Predict(_sentimentsToPredict);
            Console.WriteLine();
            Console.WriteLine("Sentiment Predictions");
            Console.WriteLine("---------------------");
            var sentimentsAndPredictions = _sentimentsToPredict.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));
            foreach (var (sentiment, prediction) in sentimentsAndPredictions)
                Console.WriteLine($"Sentiment: {sentiment.SentimentText} | Prediction: {(prediction.Sentiment ? "Positive" : "Negative")}");
        }

        public static async Task<PredictionModel<SentimentData, SentimentPrediction>> TrainModel()
        {
            var pipeline = new LearningPipeline();

            var loadData = new TextLoader(_dataPath).CreateFrom<SentimentData>();
            var features = new TextFeaturizer("Features", "SentimentText");

            pipeline.Add(loadData);
            pipeline.Add(features);

            // pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 25, NumTrees = 25, MinDocumentsInLeafs = 10 });

            var model = pipeline.Train<SentimentData, SentimentPrediction>();
            await model.WriteAsync(_modelPath);
            return model;
        }

        public static void Evaluate(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            var testData = new TextLoader(_testDataPath).CreateFrom<SentimentData>();

            var evaluator = new BinaryClassificationEvaluator();
            var metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        }
    }
}
