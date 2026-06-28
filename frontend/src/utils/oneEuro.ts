/**
 * One Euro Filter — adaptive low-pass filter for real-time signal smoothing.
 * @see https://cristal.univ-lille.fr/~casiez/1euro/
 */

class LowPassFilter {
  private y: number | null = null;

  filter(value: number, alpha: number): number {
    if (this.y === null) {
      this.y = value;
      return value;
    }
    this.y = alpha * value + (1 - alpha) * this.y;
    return this.y;
  }

  last(): number {
    return this.y ?? 0;
  }

  reset(): void {
    this.y = null;
  }
}

export class OneEuroFilter {
  private readonly xFilter = new LowPassFilter();
  private readonly dxFilter = new LowPassFilter();
  private lastTime: number | undefined;

  constructor(
    private readonly minCutoff: number,
    private readonly beta: number,
    private readonly dCutoff = 1.0,
  ) {}

  private alpha(cutoff: number, dt: number): number {
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }

  filter(value: number, timestamp: number): number {
    if (this.lastTime === undefined) {
      this.lastTime = timestamp;
      this.xFilter.filter(value, 1);
      return value;
    }

    const dt = Math.max(timestamp - this.lastTime, 1e-4);
    this.lastTime = timestamp;

    const dx = (value - this.xFilter.last()) / dt;
    const edx = this.dxFilter.filter(dx, this.alpha(this.dCutoff, dt));
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    return this.xFilter.filter(value, this.alpha(cutoff, dt));
  }

  reset(): void {
    this.xFilter.reset();
    this.dxFilter.reset();
    this.lastTime = undefined;
  }
}

type Coord = [number, number, number];

export class LandmarkSmoother {
  private readonly filters: OneEuroFilter[][];

  constructor(
    landmarkCount: number,
    minCutoff: number,
    beta: number,
  ) {
    this.filters = Array.from({ length: landmarkCount }, () =>
      Array.from({ length: 3 }, () => new OneEuroFilter(minCutoff, beta)),
    );
  }

  smooth(frame: Coord[], visibility: number[], timestamp: number): Coord[] {
    return frame.map((landmark, i) => {
      if ((visibility[i] ?? 0) <= 0) {
        for (let j = 0; j < 3; j++) {
          this.filters[i][j].reset();
        }
        return landmark;
      }
      return [
        this.filters[i][0].filter(landmark[0], timestamp),
        this.filters[i][1].filter(landmark[1], timestamp),
        this.filters[i][2].filter(landmark[2], timestamp),
      ] as Coord;
    });
  }

  reset(): void {
    for (const coordFilters of this.filters) {
      for (const filter of coordFilters) {
        filter.reset();
      }
    }
  }
}
