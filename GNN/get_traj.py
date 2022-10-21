from datetime import datetime
import numpy as np

"""code is made by Inez"""


class Worker:
  def __init__(self, id):
    self.points = []
    self.history = []
    self.id = id
    self.centroid = None
    self.prev_centroid = None

  def record(self, frame):
    self.history.append((frame, self.centroid.copy()))

  def move(self):
    self.centroid = np.mean(self.points, axis=0)

  # Given the previous and current centroid, predict the next centroid position

  def predict(self):
    self.points = []
    self.prev_centroid = self.centroid
    self.centroid += self.centroid - self.prev_centroid


def extract_trajectories(dataset):
  workers = []
  worker_idx = 1
  worker_positions = []
  attraction = 1
  worker_radius = 3
  max_occupancy = 4

  for i, entry in enumerate(dataset):
    is_analyzing = i
    remaining = entry.copy()
    row_indices, col_indices = np.nonzero(remaining)
    occupied = [[] for _ in range(len(row_indices))]

    assigned = 0
    total = np.sum(entry)

    while True:
      for w in workers:
        while len(w.points) < max_occupancy:
          # Get closest point to worker
          j = 0

          closest = None
          closest_dist = None
          closest_p = None
          while j < len(row_indices):
            p = np.float64([col_indices[j], row_indices[j]])
            if len(occupied[j]) >= entry[row_indices[j], col_indices[j]]:
              j += 1
              continue
            dist = np.linalg.norm(w.centroid - p)

            if w.id not in occupied[j]:
              if closest_dist is None or dist < closest_dist:
                closest_dist = dist
                closest = j
                closest_p = p
            j += 1

          if closest is None:
            break

          if closest_dist > worker_radius:
            break

          w.points.append(closest_p)
          w.move()
          occupied[closest].append(w.id)

          assigned += 1

      if assigned < total:
        j = 0
        while j < len(row_indices):
          if len(occupied[j]) < entry[row_indices[j], col_indices[j]]:
            p = np.float64([col_indices[j], row_indices[j]])
            worker = Worker(worker_idx)
            worker.centroid = p
            workers.append(worker)

            worker_idx += 1
            break
          j += 1

      else:
        break

    for w in workers:
      w.record(i)
      w.predict()

    # Take a snapshot of the current frame
    saved = []
    for w in workers:
      if w.centroid is not None:
        saved.append(w.centroid.copy())

    worker_positions.append(saved)

  print(f"Analysis Done! Found {worker_idx} distinct workers.")

  save_results(workers)


def save_results(workers):
  # Save results
  now = datetime.now()
  outfn = now.strftime("%m%d%Y_%H%M%S.csv")
  with open(outfn, "w") as f:
    f.write("id, frame, pos row, pos col\n")
    for w in workers:
      for (frame, (x, y)) in w.history:
        f.write(f"{w.id}, {frame}, {x}, {100-y}\n")
  print(f"Saved in {outfn}")

if __name__=="__main__":
  dataset = np.load("data/data_100cm_100p_sim1.npy")
  print(dataset.shape)

  extract_trajectories(dataset)
