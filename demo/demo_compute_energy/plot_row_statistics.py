import numpy as np
import fra_lib.funzioni as fra
import sys
print(sys.executable)
print(fra.__file__)

def main():
    cooling_rates = [0.1, 1, 10, 100]
    train_gen_energies = np.array([[-22.203766, -22.247513, -22.169745, -22.371528, -22.159061, -22.396485],
                                  [-22.096374, -22.259851, -22.116726, -22.123513, -22.260681, -22.106890],
                                  [-22.012785, -21.963463, -22.108300, -22.079973, -21.943911, -22.116610],
                                  [-22.023060, -21.861594, -22.028743, -21.922189, -21.962188, -21.962039]])
    demo_gen_energies = np.array([[-22.310905, -22.271143, -22.300758, -22.264846, -22.291715, -22.305877],
                                  [-22.295458, -22.287653, -22.276761, -22.294341, -22.282806, -22.293316],
                                  [-22.277145, -22.288354, -22.287092, -22.283929, -22.249874, -22.296120],
                                  [-22.287746, -22.271651, -22.288259, -22.296024, -22.280727, -22.256846]])
    prova = np.array([-22.251234])

    mean_train = train_gen_energies.mean(axis=1)
    mean_demo = demo_gen_energies.mean(axis=1)
    sdt_train = train_gen_energies.std(axis=1)
    sdt_demo = demo_gen_energies.std(axis=1)
    fra.plot(cooling_rates, mean_train, legenda = 'train', xlogscale = True, uy = sdt_train, incertezze = True,color = 'blue', shape = 'o-')
    fra.plot(cooling_rates, mean_demo, title = '', legenda = 'demo_gen', xlogscale = True, uy = sdt_demo, incertezze = True,
            xlabel = 'Cooling Rate',
            ylabel = 'Mean Energy', color = 'red', shape = 'o-', show=True)
if __name__ == "__main__":
    main()


