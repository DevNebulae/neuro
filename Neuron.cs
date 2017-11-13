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
    }
}
