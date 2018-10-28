using System;
using System.Collections.Generic;
using System.Text;

namespace GANLibrary
{
    class Generator
    {
        int TargetWidth = 0;
        int TargetHeight = 0;
        List<string[]> realImages = new List<string[]>();
        double fakeProb = 0.5;
        Random rand = new Random();

        private Generator() { }
        public Generator(int targetWidth, int targetHeight)
        {
            TargetWidth = targetWidth;
            TargetHeight = targetHeight;
        }

        public void AddAllReal(string[][] realImage)
        {
            foreach (string[] image in realImage)
            {
                realImages.Add(image);
            }
        }

        public void AddReal(string[] realImage)
        {
            realImages.Add(realImage);
        }

        public string[] GenerateImage()
        {
            if (rand.NextDouble() > fakeProb)
                return realImages[rand.Next(realImages.Count)];
            else
            {
                string[] toMutate = realImages[rand.Next(realImages.Count)];
                string[] toReturn = new string[toMutate.Length];

                int[,] imageVector = Utilities.StringsToVector(toMutate);
                int size = toMutate.GetLength(1);
                int[] noiseVector = GenerateNoiseVector(size);

                for (int i = 0; i < imageVector.GetLength(0); i++)
                {
                    for (int j = 0; j < imageVector.GetLength(1); j++)
                    {
                        imageVector[i, j] = imageVector[i, j] % noiseVector[j];
                    }
                }

                toReturn = Utilities.VectorToStrings(imageVector);

                return toReturn;
            }
        }

        private int[] GenerateNoiseVector(int size)
        {
            int[] noiseVector = new int[size];
            for (int i = 0; i < noiseVector.Length; i++)
            {
                noiseVector[i] = rand.Next(5);
            }
            return noiseVector;
        }
    }
}
