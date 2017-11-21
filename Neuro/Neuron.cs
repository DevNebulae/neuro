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
        private Neuron[] PreviousLayer { get; }
        /// <summary>
        /// The successors of this neuron which this neuron
        /// is connected to. When the neuron is in the last
        /// layer, the array should be null.
        /// </summary>
        private Neuron[] NextLayer { get; }
        /// <summary>
        /// The hidden gradient is the weight of the neuron
        /// that other nodes can not influence and is
        /// calculated based on the weights of the previous
        /// neurons and their outputs.
        /// </summary>
        /// <returns>
        /// a floating point number between the limits of
        /// the specified activation function.
        /// </returns>
        private double Gradient { get; set; }
        /// <summary>
        /// Represents the index of the neuron in the neural
        /// net counted from 0.
        /// </summary>
        private int Index { get; }

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

        public Neuron(Neuron[] previousLayer, Neuron[] nextLayer, int index, Random random)
        {
            this.PreviousLayer = previousLayer;
            this.NextLayer = nextLayer;
            this.Index = index;

            // Connect layer (n) with layer (n+1) if this
            // neuron is not in the last layer.
            if (nextLayer != null)
            {
                this.Connections = new NetConnection[nextLayer.Length];

                // When this node is not the last node, add
                // all nodes as a connection with a
                // randomized weight.
                for (int i = 0; i < nextLayer.Length; i++)
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
        /// <item>
        ///     <term>Logistic sigmoid</term>
        ///     <description>
        ///     A decimal value between 0 and 1;
        ///     </description>
        /// </item>
        /// <item>
        ///     <term>Hyperbolic tangent</term>
        ///     <description>
        ///     A decimal value between -1 and +1;
        ///     </description>
        /// </item>
        /// <item>
        ///     <term>Heaviside step</term>
        ///     <description>
        ///     A boolean value between of either 0 or 1;
        ///     </description>
        /// </item>
        /// <item>
        ///     <term>Softmax</term>
        ///     <description>
        ///     A decimal value between 0 and 1 where the
        ///     sum of all weights equals to 1.
        ///     </description>
        /// </item>
        /// </list>
        /// </summary>
        /// <param name="value"></param>
        /// <returns>
        /// the value of a hyperbolic tangent function.
        /// </returns>
        public static double Activation(double value) => Math.Tanh(value);

        /// <summary>
        /// Derives the activation function, which in this
        /// case is equal to 1 - tanh(x)^2.
        /// </summary>
        /// <param name="value"></param>
        /// <returns>
        /// the derived value of the activation function.
        /// </returns>
        public static double ActivationDerivative(double value) => 1 - Math.Pow(value, 2);

        /// <summary>
        /// Calculate the hidden gradient based on the sum
        /// of the weight to the nodes in the next layer 
        /// multiplied by the hidden gradient of those
        /// nodes.
        /// </summary>
        public void CalculateHiddenGradient()
        {
            double dow = this.SumDOW();
            this.Gradient = dow * Neuron.ActivationDerivative(this.Output);
        }

        public void CalculateOutputGradient(double target)
        {
            double delta = target - this.Output;
            this.Gradient = delta * Neuron.ActivationDerivative(this.Output);
        }

        /// <summary>
        /// Calculate the current output property based on
        /// the output values multiplied by the weight of
        /// the connection of the predecessing neuron(s).
        /// </summary>
        public void FeedForward()
        {
            double value = 0;

            Array.ForEach(this.PreviousLayer, neuron =>
            {
                value += neuron.Output * neuron.Connections[this.Index].Weight;
            });

            this.Output = Neuron.Activation(value);
        }

        /// <summary>
        /// Calculate the sum of the connections' weight to
        /// the neurons in the next layer multiplied by the
        /// hidden gradient of those nodes.
        /// </summary>
        /// <returns>
        /// a floating point number representing the sum of.
        /// </returns>
        public double SumDOW()
        {
            double dow = 0;

            // The last neuron is skipped in the next layer,
            // because it is a bias neuron.
            // TODO: Figure out how to specify that the last
            // neuron is a biased neuron or that document
            // that it is by default.
            for (int i = 0; i < this.NextLayer.Length - 1; i++)
            {
                dow += this.Connections[i].Weight * this.NextLayer[i].Gradient;
            }

            return dow;
        }

        public void UpdateInputWeights() =>
            Array.ForEach(this.PreviousLayer, neuron =>
            {
                double oldDeltaWeight = neuron.Connections[this.Index].DeltaWeight;
                double newDeltaWeight = Neuron.ETA * neuron.Output * this.Gradient + Neuron.ALPHA * oldDeltaWeight;

                neuron.Connections[this.Index].DeltaWeight = newDeltaWeight;
                neuron.Connections[this.Index].Weight += newDeltaWeight;
            });
    }
}
