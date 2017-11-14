using System;

namespace Neuro
{
    public class NeuralNetwork
    {
        private Neuron[][] Layers { get; }
        private Random random = new Random();
        public double Error { get; private set; }

        /// <summary>
        /// Initialize a neural network with the amount of
        /// nodes specified in each layer based on the
        /// specified topology.
        /// </summary>
        /// <param name="topology">An array of integers
        /// specifying what the neural network should look 
        /// like. For example, if the topology is [2, 3, 1],
        /// 2 nodes will be created in the first layer, 3 in
        /// the second, et cetera.</param>
        public NeuralNetwork(int[] topology)
        {
            this.Layers = new Neuron[topology.Length][];

            for (int layerIndex = 0; layerIndex < topology.Length; layerIndex++)
            {
                // Always add one more neuron than
                // specified, because of the bias neuron.
                this.Layers[layerIndex] = new Neuron[topology[layerIndex] + 1];
            }

            Neuron[] previousLayer = null;
            Neuron[] nextLayer;

            for (int layerIndex = 0; layerIndex < topology.Length; layerIndex++)
            {
                // If the layer count is 0, then the neurons
                // in that layer do not have a previous
                // layer. If that is not the case, select
                // the last neuron layer.
                if (layerIndex > 0) previousLayer = this.Layers[layerIndex - 1];

                nextLayer = null;

                // If the layer count is at the end of the
                // topology, then the neurons in that layer
                // do not have a successing layer. If that
                // is not the case, select the next layer.
                if (layerIndex < topology.Length - 1) nextLayer = this.Layers[layerIndex + 1];

                for (int neuronIndex = 0; neuronIndex < topology[layerIndex] + 1; neuronIndex++)
                {
                    this.Layers[layerIndex][neuronIndex] = new Neuron(previousLayer, nextLayer, neuronIndex, random);
                }

                // Add a bias neuron at the end of each
                // layer
                Neuron bias = this.Layers[layerIndex][topology[layerIndex]];
                bias.Output = 1;
            }
        }

        public void BackPropagate(double[] targetValues)
        {
            Neuron[] outputLayer = this.Layers[this.Layers.Length - 1];

            if (targetValues.Length != outputLayer.Length - 1) // take bias nueron into account
            {
                throw new IndexOutOfRangeException("invalid number of targets");
            }

            this.Error = RMSNetError(targetValues);

            // Calculate a recent average error
            //recentAvgError = (recentAvgError * recentAvgSmoothingFactor + error) / (recentAvgSmoothingFactor + 1.0);

            // Calc output gradient
            for (int i = 0; i < targetValues.Length; i++)
            {
                outputLayer[i].CalculateOutputGradient(targetValues[i]);
            }

            // Calc hidden layer gradients
            for (int i = this.Layers.Length - 2; i > 0; i--)
            {
                for (int j = 0; j < this.Layers[i].Length; j++)
                {
                    this.Layers[i][j].CalculateHiddenGradient();
                }
            }

            // Update connection weights
            for (int i = this.Layers.Length - 1; i > 0; i--)
            {
                for (int j = 0; j < this.Layers[i].Length - 1; j++)
                {
                    this.Layers[i][j].UpdateInputWeights();
                }
            }
        }


        public void FeedForward(double[] inputValues)
        {
            Neuron[] inputLayer = this.Layers[0];

            if (inputValues.Length != inputLayer.Length - 1)
            {
                throw new IndexOutOfRangeException($"The number of input values did not match the amount of nodes in the neural network. I got {inputValues.Length} inputs, while the size of the neural network layer is {inputLayer.Length - 1}.");
            }

            for (int i = 0; i < inputValues.Length; i++)
            {
                inputLayer[i].Output = inputValues[i];
            }

            for (int layerNum = 1; layerNum < this.Layers.Length; layerNum++)
            {
                for (int n = 0; n < this.Layers[layerNum].Length - 1; n++)  // do not feedForward the bias neuron
                {
                    this.Layers[layerNum][n].FeedForward();
                }
            }
        }
        public double RMSNetError(double[] targetValues)
        {
            double error = 0.0;
            Neuron[] outputLayer = this.Layers[this.Layers.Length - 1];

            // calculate Root Mean Square net error
            for (int i = 0; i < targetValues.Length; i++)
            {
                double delta = targetValues[i] - outputLayer[i].Output;
                error += delta * delta;
            }

            error = Math.Sqrt(error / targetValues.Length);

            return error;
        }
    }
}