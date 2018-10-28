using System;
using System.IO;

namespace GANLibrary
{
    public class GAN
    {
        Generator gen = null;
        Discriminator dis = null;
        Random rand = new Random();


        public GAN(string ganToLoad = "")
        {
        }

        public void LoadReal(string[] realCopies)
        {
            foreach (string real in realCopies)
            {
                StreamReader reader = File.Exists(real) ? new StreamReader(real) : null;
                if (reader != null)
                {
                    while (!reader.EndOfStream)
                    {

                    }
                }
            }
        }
    }
}
