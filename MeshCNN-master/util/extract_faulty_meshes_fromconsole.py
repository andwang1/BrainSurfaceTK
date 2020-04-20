# with open("lognative10k", "r") as f:
#     log = [line.strip() for line in f.readlines()]
#
# faulty_meshes = []
#
# for i in range(len(log)):
#     if log[i] =='HERE':
#         x = log[i]
#         element = i
#         while x == 'HERE':
#             element -= 1
#             x = log[element]
#
#         faulty_meshes.append(x)
#
# print(faulty_meshes)
# print(set(faulty_meshes))

# print(len(set(faulty_meshes)))


with open("logmerged10k.txt", "r") as f:
    log = [line.strip() for line in f.readlines()]

faulty_meshes = []

for i in range(len(log)):
    if log[i] =='HERE':
        x = log[i]
        element = i
        while x == 'HERE':
            element -= 1
            x = log[element]

        faulty_meshes.append(x)

print(faulty_meshes)
print(set(faulty_meshes))

print(len(set(faulty_meshes)))

