#include <algorithm>
#include <iostream>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <cstdlib>


using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::set;
using std::vector;


class DisjointSetUnion {
public:
    explicit DisjointSetUnion(size_t size) : parent_(), ranks_(size, 0) {
        parent_.reserve(size);
        for (size_t i = 0; i < size; ++i) {
            parent_.push_back(i);
        }
    }

    size_t find(size_t node) {
        if (parent_[node] != node) {
            parent_[node] = find(parent_[node]);
        }
        return parent_[node];
    }

    void union_sets(size_t first, size_t second) {
        size_t first_root = find(first);
        size_t second_root = find(second);
        if (first_root == second_root) {
            return;
        }
        if (ranks_[first_root] < ranks_[second_root]) {
            parent_[first_root] = second_root;
        } else if (ranks_[first_root] > ranks_[second_root]) {
            parent_[second_root] = first_root;
        } else {
            parent_[second_root] = first_root;
            ++ranks_[first_root];
        }
    }

private:
    vector<size_t> parent_;
    vector<size_t> ranks_;
};

struct Edge {
    size_t from;
    size_t to;
    double weight;
};

struct Point2D {
    double x, y;
};

void RenumerateLabels(vector<size_t> &rawLabels) {
    vector<int> rawToNew(rawLabels.size(), -1);
    size_t indexesUsed = 0;
    for (size_t i = 0; i < rawLabels.size(); ++i) {
        size_t oldLabel = rawLabels[i];
        if (rawToNew[oldLabel] == -1) {
            rawToNew[oldLabel] = indexesUsed;
            ++indexesUsed;
        }
        rawLabels[i] = rawToNew[oldLabel];
    }
}

vector<size_t> ClusterGraphMST(vector<Edge> edges, size_t vertexCount, size_t clusterCount) {
    DisjointSetUnion sets(vertexCount);
    std::sort(edges.begin(), edges.end(), [](Edge a, Edge b) { return a.weight < b.weight; });
    size_t counter = 0;
    for (auto edge : edges) {
        if (sets.find(edge.from) != sets.find(edge.to)) {
            sets.union_sets(edge.from, edge.to);
            if (++counter == vertexCount - clusterCount) {
                break;
            }
        }
    }
    vector<size_t> clusters(vertexCount);
    for (size_t vertex = 0; vertex < vertexCount; ++vertex) {
        clusters[vertex] = sets.find(vertex);
    }
    RenumerateLabels(clusters);
    return clusters;
}


template<typename T, typename Dist>
vector<Edge> PairwiseDistances(vector<T> objects, Dist distance) {
    vector<Edge> edges;
    for (size_t i = 0; i < objects.size(); ++i) {
        for (size_t j = i + 1; j < objects.size(); ++j) {
            edges.push_back({i, j, distance(objects[i], objects[j])});
        }
    }
    return edges;
}

template<typename T, typename Dist>
vector<size_t> ClusterMST(const vector<T> &objects, Dist distance, size_t clusterCount) {
    vector<Edge> edges = PairwiseDistances(objects, distance);
    return ClusterGraphMST(edges, objects.size(), clusterCount);
}

template<typename T>
void recalculate_centers(vector<size_t>& clusters, vector<T>& objects, vector<Point2D>& centers) {
    vector<size_t> point_counts(clusters.size());
    centers.assign(clusters.size(), {0, 0});
    for (size_t i = 0; i < objects.size(); ++i) {
        ++point_counts[clusters[i]];
        centers[clusters[i]].x += objects[i].x;
        centers[clusters[i]].y += objects[i].y;
    }
    for (size_t i = 0; i < clusters.size(); ++i) {
        centers[i].x /= point_counts[i];
        centers[i].y /= point_counts[i];
    }
}

template<typename T, typename Dist>
vector<size_t> ClusterMinDistToCenter(vector<T> &objects, Dist distance, size_t clusterCount) {
    vector<size_t> clusters(objects.size());
    vector<Point2D> centers;
    set<Point2D> centrs;
    while (centers.size() < clusterCount) {
        size_t new_center = rand() % objects.size();
        centers.push_back(objects[new_center]);
    }

    bool changed;
    do {
        changed = false;
        for (size_t i = 0; i < objects.size(); ++i) {
            size_t cluster = clusters[i];
            for (size_t j = 0; j < centers.size(); ++j) {
                if (distance(centers[cluster], objects[i]) > distance(centers[j], objects[i]) && cluster != j) {
                    cluster = j;
                    changed = true;
                }
            }
            clusters[i] = cluster;
        }
        if (changed) {
            recalculate_centers(clusters, objects, centers);
        }
    } while (changed);
    return clusters;
}


double EuclidianDistance(const Point2D &first, const Point2D &second) {
    return std::sqrt((first.x - second.x) * (first.x - second.x) + (first.y - second.y) * (first.y - second.y));
}


vector<Point2D> Random2DClusters(const vector<Point2D> &centers, const vector<double> &xVariances,
                                 const vector<double> &yVariances, size_t pointsCount) {
    auto baseGenerator = std::default_random_engine();
    auto generateCluster = std::uniform_int_distribution<size_t>(0, centers.size() - 1);
    auto generateDeviation = std::normal_distribution<double>();

    vector<Point2D> results;
    for (size_t i = 0; i < pointsCount; ++i) {
        size_t c = generateCluster(baseGenerator);
        double x = centers[c].x + generateDeviation(baseGenerator) * xVariances[c];
        double y = centers[c].y + generateDeviation(baseGenerator) * yVariances[c];
        results.push_back({x, y});
    }

    return results;
}

void GNUPlotClusters2D(vector<Point2D> &points, const vector<size_t> &labels, size_t clustersCount,
                       const string &outFolder) {
    std::ofstream scriptOut(outFolder + "/script.txt");
    scriptOut << "set term png;\nset output \"plot.png\"\n";
    scriptOut << "plot ";
    for (size_t cluster = 0; cluster < clustersCount; ++cluster) {
        string filename = std::to_string(cluster) + ".dat";
        std::ofstream fileOut(outFolder + "/" + filename);
        scriptOut << "\"" << filename << "\"" << " with points, ";

        for (size_t i = 0; i < points.size(); ++i) {
            if (labels[i] == cluster) {
                fileOut << points[i].x << "\t" << points[i].y << "\n";
            }
        }
    }
}


int main() {
    std::vector<Point2D> centers{{0, 0}, {6, -1}, {5, 6}, {5, -1}, {2, 3}};
    std::vector<double> xVariances{0.4, 0.3, 0.3, 0.4, 0.3, 0.4};
    std::vector<double> yVariances{0.5, 0.5, 0.3, 0.5, 0.4, 0.4};
    auto points = Random2DClusters(centers, xVariances, yVariances, 1000);

    const size_t kClusterCount = 3;
    vector<size_t> labels(points.size(), 0);
    GNUPlotClusters2D(points, labels, 1, "plot_base");

    labels = ClusterMST(points, EuclidianDistance, kClusterCount);
    GNUPlotClusters2D(points, labels, kClusterCount, "plot_mst");

    labels = ClusterMinDistToCenter(points, EuclidianDistance, kClusterCount);
    GNUPlotClusters2D(points, labels, kClusterCount, "plot_mdc");

    return 0;
}
