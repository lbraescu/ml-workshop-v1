using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using api.Models;
using Microsoft.ML;

namespace api.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class PredictController : ControllerBase
    {
        [HttpGet]
        public string Get(string text)
        {
            var model = PredictionModel.ReadAsync<SentimentData, SentimentPrediction>("Model.zip").Result;
            var prediction = model.Predict(new SentimentData { SentimentText = text });
            return prediction.Sentiment ? "Positive" : "Negative";
        }
    }
}