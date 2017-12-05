using Neuro;
using System;
using System.IO;
using System.Linq;

namespace NeuroConsole
{
    public class Program
    {
        static void Train(NeuralNetwork network, double[] inputs, double[] targets)
        {
            network.FeedForward(inputs);
            network.BackPropagate(targets);
        }

        static void Main(string[] args)
        {
            var trainSize = 5000;
            var testSize = 10000;

            var network = new NeuralNetwork(new int[] { 784, 300, 10 });

            Console.WriteLine("Start reading training- and testing images and labels");

            byte[] trainingLabels = MNISTReader.ReadMNISTLabels(trainSize, @"assets/train-labels.idx1-ubyte");
            byte[][][] trainImages = MNISTReader.ReadMNISTImages(trainSize, 28, @"assets/train-images.idx3-ubyte");

            Console.WriteLine("Done reading training images and labels");

            var correct = 0;
            var incorrect = 0;

            byte[] testLabels = MNISTReader.ReadMNISTLabels(testSize, @"assets/t10k-labels.idx1-ubyte");
            byte[][][] testImages = MNISTReader.ReadMNISTImages(testSize, 28, @"assets/t10k-images.idx3-ubyte");

            Console.WriteLine("Done reading testing images and labels");

            for (int i = 0; i < trainImages.Length; i++)
            {
                var digitImage = new DigitImage(trainImages[i], trainingLabels[i]);
                var inputs = digitImage.Flatten().Select(x => DigitImage.Normalize(x)).ToArray();
                var targets = digitImage.Output();

                Program.Train(network, inputs, targets);

                if ((i % 1000 == 0 && i != 0) || (i == trainImages.Length - 1))
                    Console.WriteLine($"I just finished iteration {i.ToString()}");
            }

            Console.WriteLine("Done training the neural network");
            Console.WriteLine("Start testing of number recognition");

            for (int i = 0; i < testImages.Length; i++)
            {
                var digitImage = new DigitImage(testImages[i], testLabels[i]);
                var inputs = digitImage.Flatten().Select(x => DigitImage.Normalize(x)).ToArray();
                var target = (int)digitImage.label;

                network.FeedForward(inputs);
                var outputs = network.Results();

                Console.WriteLine($"{outputs[target]}, {outputs.Max()}");

                if (outputs[target] == outputs.Max())
                    correct++;
                else
                    incorrect++;
            }

            Console.WriteLine($"Testing score of {(double)correct / (double)testSize}%. I recognized {correct} of {testSize} images correctly, while I failed to recognize {incorrect} of {testSize} images.");
        }
    }
}
