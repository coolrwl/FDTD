import pylab
import numpy as np
import json
import pathlib

A = 0


def f(x):
    return 0.5 + (
        np.sin(x ** 2 - A ** 2) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + A ** 2))


x_min = -10
x_max = 10
dx = 0.001

x = np.arange(x_min, x_max+dx, dx)
y = f(x)

res = {
    "x": x.tolist(),
    "y": y.tolist(),
}

path = pathlib.Path("results")  #извлечем директорию "results"
path.mkdir(exist_ok=True)   #создаем новый каталог по заданному пути, exist_ok=True, то исключения FileExistsError будут игнорироваться(поднимается при попытке создать файл или каталог, который уже существует)
file = path / "result_task1.json"   #создать файл под названием "result_task1.json" в текущем рабочем каталоге
out = file.open("w")    #открыть файл для записи
json.dump(res, out)   #конвертация в тип .jason и запись в него res
out.close()


pylab.plot(x, y)
pylab.grid()
pylab.savefig("results/task1.png")  #сохранить график как изображение
pylab.show()