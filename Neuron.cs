using System;

namespace Neuro
{
    public class Neuron
    {
        /// <summary>
        /// A neural net is built up from left to right
        /// where a neuron is connected with its 
        /// predecessors and its successors.
        /// </summary>
        public NetConnection[] Connections { get; }
        /// <summary>
        /// The predecessors of this neuron which are
        /// connected to this neuron. When the neuron is in
        /// the first layer, the array should be null.
        /// </summary>
        private Neuron[] previousLayers;
        /// <summary>
        /// The successors of this neuron which this neuron
        /// is connected to. When the neuron is in the last
        /// layer, the array should be null.
        /// </summary>
        private Neuron[] nextLayers;
        private double gradient;
        /// <summary>
        /// Represents the index of the neuron in the neural
        /// net counted from 0.
        /// </summary>
        private int index;

        /// <summary>
        /// The ETA is the rate at which the neural net 
        /// learns, whereby 0 indicates a very slow learner,
        /// 0.2 indicates a moderate learner and 1.0
        /// indicates a reckless learner.
        /// </summary>
        private static double ETA = 0.15;
        /// <summary>
        /// The alpha determines the momentum of the neural
        /// net where 0 indicates no momentum and 1
        /// indicates the upper limit in momentum.
        /// </summary>
        private static double ALPHA = 0.5;

        public double Output { get; set; }

        public Neuron(Neuron[] previousLayers, Neuron[] nextLayers, int index, Random random)
        {
            this.previousLayers = previousLayers;
            this.nextLayers = nextLayers;
            this.index = index;

            // Connect layer (n) with layer (n+1) if this
            // neuron is not in the last layer.
            if (nextLayers != null)
            {
                this.Connections = new NetConnection[nextLayers.Length];

                // When this node is not the last node, add
                // all nodes as a connection with a
                // randomized weight.
                for (int i = 0; i < nextLayers.Length; i++)
                {
                    this.Connections[i].Weight = random.NextDouble();
                    this.Connections[i].DeltaWeight = 0;
                }
            }
            else
            {
                this.Connections = new NetConnection[0];
            }
        }

        /// <summary>
        /// In neural networks, the activation functions
        /// transforms its current values to a more
        /// meaningful value based on the activation
        /// function that you use. The four most well-known 
        /// activation functions are:
        /// <list type="bullet">
        /// <item><description>Logistic sigmoid: a decimal value between 0 and 1;</description></item>
        /// <item><description>Hyperbolic tangent: a decimal value between -1 and +1;</description></item>
        /// <item><description>Heaviside step: a boolean value between of either 0 or 1;</description></item>
        /// <item><description>Softmax: a decimal value between 0 and 1 where the sum of all weights equals to 1.</description></item>
        /// </list>
        /// </summary>
        /// <param name="value"></param>
        /// <returns>the value of a logistic sigmoid function.</returns>
        public static double Activation(double value) => Math.Tanh(value);

        public static double ActivationDerivative(double value) => 1 - Math.Pow(value, 2);

        public void CalculateOutputGradient(double target)
        {
            double delta = target - this.Output;
            this.gradient = delta * Neuron.ActivationDerivative(this.Output);
        }

        public void FeedForward()
        {
            double value = 0;

            Array.ForEach(this.previousLayers, neuron =>
            {
                value += neuron.Output * neuron.Connections[this.index].Weight;
            });

            this.Output = Neuron.Activation(value);
        }

        public void UpdateInputWeights() =>
            Array.ForEach(this.previousLayers, neuron =>
            {
                double oldDeltaWeight = neuron.Connections[this.index].DeltaWeight;
                double newDeltaWeight = Neuron.ETA * neuron.Output * this.gradient + Neuron.ALPHA * oldDeltaWeight;

                neuron.Connections[this.index].DeltaWeight = newDeltaWeight;
                neuron.Connections[this.index].Weight += newDeltaWeight;
            });
    }
}
