using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace regression
{
    public class Program
    {
        private static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        private static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        private static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        private static readonly TaxiTrip _tripToPredict = new TaxiTrip
        {
            VendorId = "VTS",
            RateCode = "1",
            PassengerCount = 1,
            TripDistance = 10.33f,
            PaymentType = "CSH",
            FareAmount = 0 //actual = 29.5
        };

        public static async Task Main(string[] args)
        {
            var model = await TrainModel();

            Evaluate(model);

            var prediction = model.Predict(_tripToPredict);
            Console.WriteLine($"Predicted fare: {prediction.FareAmount}");

            Console.ReadKey();
        }

        public static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> TrainModel()
        {
            var pipeline = new LearningPipeline();

            var loadData = new TextLoader(_dataPath).CreateFrom<TaxiTrip>(useHeader: true, separator: ',');
            var copyLabels = new ColumnCopier(("FareAmount", "Label"));
            var convertToNumeric = new CategoricalOneHotVectorizer("VendorId", "RateCode", "PaymentType");
            var features = new ColumnConcatenator("Features",
                                    "VendorId", "RateCode", "PassengerCount", "TripDistance", "PaymentType");

            pipeline.Add(loadData);
            pipeline.Add(copyLabels);
            pipeline.Add(convertToNumeric);
            pipeline.Add(features);

            pipeline.Add(new FastTreeRegressor());

            var model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();
            await model.WriteAsync(_modelPath);
            return model;
        }

        private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            var testData = new TextLoader(_testDataPath).CreateFrom<TaxiTrip>(useHeader: true, separator: ',');

            var evaluator = new RegressionEvaluator();
            var metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine($"Rms = {metrics.Rms}");
            Console.WriteLine($"RSquared = {metrics.RSquared}");
        }
    }
}
