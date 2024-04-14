﻿using System;
using RestaurantSentiment.ML;

namespace RestaurantSentiment
{
    class Program
    {
        public static void Main(String[] args)
        {
            if(args.Length!=2 ) {


                Console.WriteLine($"Invalid arguments passed in, exiting.{Environment.NewLine}{Environment.NewLine}Usage:{Environment.NewLine}" +
                                  $"predict <sentence of text to predict against>{Environment.NewLine}" +
                                  $"or {Environment.NewLine}" +
                                  $"train <path to training data file>{Environment.NewLine}");

                return;
            
                   
            
            }
            switch(args[0])
            {
                case "predict":
                    new Predictor().Predict(args[1]); 
                    break;
                case "train":
                    new Trainer().Train(args[1]); 
                    break;
                default:
                    Console.WriteLine($"{args[0]} is an invalid option");
                    break;
            }
        }
    }



}