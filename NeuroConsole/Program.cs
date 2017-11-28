using System;
using System.IO;

namespace NeuroConsole
{
    public class Program
    {
        static void Main(string[] args)
        {
            byte[] labels = MNISTReader.ReadMNISTLabels(10000, @"assets/t10k-labels.idx1-ubyte");
            byte[][][] images = MNISTReader.ReadMNISTImages(10000, 28, @"assets/t10k-images.idx3-ubyte");

            for (int i = 0; i < images.Length; i++) {
                var digitImage = new DigitImage(images[i], labels[i]);
                Console.WriteLine(digitImage.ToString());
            }
        }
    } 
}
