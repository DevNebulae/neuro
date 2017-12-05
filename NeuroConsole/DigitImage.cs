using System;
using System.Linq;

namespace NeuroConsole
{
    public class DigitImage
    {
        public byte[][] pixels;
        public byte label;

        public DigitImage(byte[][] pixels,
          byte label)
        {
            this.pixels = new byte[28][];
            for (int i = 0; i < this.pixels.Length; ++i)
                this.pixels[i] = new byte[28];

            for (int i = 0; i < 28; ++i)
                for (int j = 0; j < 28; ++j)
                    this.pixels[i][j] = pixels[i][j];

            this.label = label;
        }

        public int[] Flatten() => this.pixels.SelectMany(x => x).Select(x => (int)x).ToArray();

        public static double Normalize(int value) => (double)value / 255.0;

        public double[] Output()
        {
            var label = (int)this.label;
            var output = new double[10];

            for (int i = 0; i < 10; i++)
            {
                if (i != label)
                    output[i] = 0;
                else
                    output[i] = 1;
            }

            return output;
        }

        public override string ToString()
        {
            string s = "";
            for (int i = 0; i < 28; ++i)
            {
                for (int j = 0; j < 28; ++j)
                {
                    Console.WriteLine(this.pixels[i][j]);
                    if (this.pixels[i][j] == 0)
                        s += " "; // white
                    else if (this.pixels[i][j] == 255)
                        s += "O"; // black
                    else
                        s += "."; // gray
                }
                s += "\n";
            }
            s += this.label.ToString();
            return s;
        } // ToString

    }
}