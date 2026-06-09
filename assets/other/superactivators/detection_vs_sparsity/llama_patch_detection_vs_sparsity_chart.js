(function () {
  function injectStylesOnce() {
    if (document.getElementById("compare-methods-chart-styles")) {
      return;
    }

    const style = document.createElement("style");
    style.id = "compare-methods-chart-styles";
    style.textContent = `
      .compare-methods-root {
        width: 100%;
      }
      .compare-methods-legend {
        display: flex;
        justify-content: center;
        gap: 1rem 1.4rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
        font: 600 14px Verdana, Geneva, sans-serif;
        color: #111111;
      }
      .compare-methods-legend-item {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
      }
      .compare-methods-legend-line {
        width: 24px;
        border-top-width: 3px;
        border-top-style: solid;
        transform: translateY(-1px);
      }
      .compare-methods-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
      }
      .compare-methods-panel {
        min-height: 290px;
      }
      @media (max-width: 1100px) {
        .compare-methods-grid {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }
      }
      @media (max-width: 760px) {
        .compare-methods-grid {
          grid-template-columns: 1fr;
        }
        .compare-methods-panel {
          min-height: 260px;
        }
      }
    `;
    document.head.appendChild(style);
  }

  function tickLabel(value) {
    const rounded = Math.round(Number(value) * 100);
    return [10, 30, 50, 70, 90].includes(rounded) ? rounded + "%" : "";
  }

  function buildLegend(root, chartData) {
    const legend = document.createElement("div");
    legend.className = "compare-methods-legend";

    chartData.series.forEach((series) => {
      const item = document.createElement("div");
      item.className = "compare-methods-legend-item";

      const line = document.createElement("span");
      line.className = "compare-methods-legend-line";
      line.style.borderTopColor = series.color;
      if (series.borderDash && series.borderDash.length) {
        line.style.borderTopStyle = "dashed";
      }

      const label = document.createElement("span");
      label.textContent = series.label;

      item.appendChild(line);
      item.appendChild(label);
      legend.appendChild(item);
    });

    root.appendChild(legend);
  }

  function buildGrid(root, charts) {
    const grid = document.createElement("div");
    grid.className = "compare-methods-grid";

    const canvases = charts.map(() => {
      const panel = document.createElement("div");
      panel.className = "compare-methods-panel";

      const canvas = document.createElement("canvas");
      panel.appendChild(canvas);
      grid.appendChild(panel);
      return canvas;
    });

    root.appendChild(grid);
    return canvases;
  }

  function formatYTick(value, yMax) {
    return yMax <= 0.6 ? Number(value).toFixed(2) : Number(value).toFixed(1);
  }

  function buildConfig(chartData) {
    return {
      type: "line",
      data: {
        datasets: chartData.series.map((series) => ({
          label: series.label,
          data: series.points,
          parsing: false,
          borderColor: series.color,
          backgroundColor: series.color,
          borderWidth: 2.5,
          borderDash: series.borderDash || [],
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHitRadius: 10,
          tension: 0,
          fill: false
        }))
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { display: false },
          title: {
            display: true,
            text: chartData.title,
            color: "#111111",
            font: {
              family: "Verdana, Geneva, sans-serif",
              size: 17,
              weight: "700"
            },
            padding: { bottom: 10 }
          },
          tooltip: {
            callbacks: {
              label(context) {
                return context.dataset.label + ": " + Number(context.parsed.y).toFixed(3);
              }
            }
          }
        },
        scales: {
          x: {
            type: "linear",
            min: 0,
            max: 1,
            ticks: {
              stepSize: 0.1,
              color: "#222222",
              callback: tickLabel,
              font: {
                family: "Verdana, Geneva, sans-serif",
                size: 12
              }
            },
            title: {
              display: true,
              text: "Sparsity Level (\u03b4)",
              color: "#222222",
              font: {
                family: "Verdana, Geneva, sans-serif",
                size: 13,
                weight: "600"
              }
            },
            grid: {
              display: false
            },
            border: {
              display: false
            }
          },
          y: {
            min: chartData.yMin,
            max: chartData.yMax,
            ticks: {
              color: "#222222",
              maxTicksLimit: 6,
              callback(value) {
                return Number(value) === 0 ? "" : formatYTick(value, chartData.yMax);
              },
              font: {
                family: "Verdana, Geneva, sans-serif",
                size: 11
              }
            },
            title: {
              display: true,
              text: "Avg Detection F1",
              color: "#222222",
              font: {
                family: "Verdana, Geneva, sans-serif",
                size: 13,
                weight: "600"
              }
            },
            grid: {
              color: "rgba(0, 0, 0, 0.18)",
              borderDash: [4, 4],
              lineWidth: 1
            },
            border: {
              display: false
            }
          }
        }
      }
    };
  }

  function renderCompareMethodsCharts(target, bundle) {
    if (!window.Chart) {
      throw new Error("Chart.js must be loaded before rendering compare-methods charts.");
    }

    const root =
      typeof target === "string" ? document.querySelector(target) : target;
    if (!root) {
      throw new Error("Could not find target element for compare-methods charts.");
    }

    injectStylesOnce();
    root.innerHTML = "";
    root.classList.add("compare-methods-root");

    buildLegend(root, bundle.charts[0]);
    const canvases = buildGrid(root, bundle.charts);

    return canvases.map((canvas, index) => {
      const chartData = bundle.charts[index];
      return new window.Chart(canvas, buildConfig(chartData));
    });
  }

  window.renderCompareMethodsCharts = renderCompareMethodsCharts;
})();
