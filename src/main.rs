//! Quantum-Classical 3D Corridor Pathfinder
//!
//! Phase 1 – Classical: A* on a 3D cost grid     (pathfinding + ndarray)
//! Phase 2 – Quantum:   QUBO route refinement    (quantrs2-tytan SASampler)
//! Phase 3 – Summary:   cost comparison & report

use ndarray::{Array2, Array3};
use pathfinding::prelude::astar;
use quantrs2_tytan::sampler::{SASampler, Sampler};
use std::collections::HashMap;

type Pos = (usize, usize, usize);

/// Side length of the cubic cost grid.
/// Change to 100 for a 100³ production run (A* will be slower).
const GRID: usize = 20;

/// Length of the path sub-window fed into the QUBO.
const WINDOW: usize = 5;

/// Maximum number of alternative routes to enumerate for the QUBO.
const MAX_ALTS: usize = 12;

// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║   Quantum-Classical 3D Corridor Pathfinder       ║");
    println!("╚══════════════════════════════════════════════════╝\n");

    // ── Phase 1: Classical A* ────────────────────────────────────────────────
    println!("Phase 1 ▸ Classical A* on a {}³ cost grid", GRID);

    let grid = build_terrain();
    let start: Pos = (0, 0, 0);
    let goal: Pos = (GRID - 1, GRID - 1, GRID - 1);

    let (path, classical_cost) = find_path_astar(&grid, start, goal);

    println!("  nodes in path : {}", path.len());
    println!("  total cost    : {}", classical_cost);

    // ── Phase 2: Quantum QUBO refinement ────────────────────────────────────
    println!("\nPhase 2 ▸ QUBO route refinement (quantrs2-tytan SASampler)");

    let mid = path.len() / 2;
    let sub_end = (mid + WINDOW).min(path.len() - 1);
    let from = path[mid];
    let to = path[sub_end];

    println!("  window        : path[{}..{}]", mid, sub_end);
    println!("  from {:?}  →  to {:?}", from, to);

    // Enumerate alternative routes through this window (includes the A* route)
    let mut alternatives = enumerate_paths(&grid, from, to, WINDOW + 2, MAX_ALTS - 1);

    // Always include the original A* sub-segment as the first alternative
    let astar_sub: Vec<Pos> = path[mid..=sub_end].to_vec();
    let astar_sub_cost: u32 = astar_sub[1..].iter().map(|&p| grid[[p.0, p.1, p.2]]).sum();
    alternatives.insert(0, (astar_sub, astar_sub_cost));
    // Remove exact duplicates of the A* route that DFS might have also found
    alternatives.dedup_by_key(|(_, c)| *c);

    println!("  alternatives  : {} routes (incl. A* baseline)", alternatives.len());

    for (i, (_, cost)) in alternatives.iter().enumerate() {
        let marker = if i == 0 { " ← A* baseline" } else { "" };
        println!("    r{i}: cost {cost}{marker}");
    }

    if alternatives.len() < 2 {
        println!("  Only one route found — nothing for QUBO to optimise.");
    } else {
        // Build and solve the QUBO
        let (qubo_matrix, var_map) = build_one_hot_qubo(&alternatives);

        let sampler = SASampler::new(Some(42));
        let results = sampler
            .run_qubo(&(qubo_matrix, var_map), 300)
            .expect("SASampler failed");

        let best = &results[0];
        println!("\n  QUBO result");
        println!("  best energy   : {:.3}", best.energy);

        // Identify which route index the QUBO selected
        let selected_idx = (0..alternatives.len())
            .find(|&i| *best.assignments.get(&format!("r{i}")).unwrap_or(&false))
            .unwrap_or(0);

        let (_, selected_cost) = &alternatives[selected_idx];
        println!("  selected      : r{selected_idx} (cost {selected_cost})");

        if *selected_cost < astar_sub_cost {
            let saving = astar_sub_cost - selected_cost;
            println!(
                "  ✓ QUBO found a cheaper sub-route! Saving: {} ({:.1}% of sub-segment)",
                saving,
                100.0 * saving as f64 / astar_sub_cost as f64
            );
        } else {
            println!("  A* sub-segment is already optimal — QUBO confirms it.");
        }
    }

    // ── Phase 3: Summary ────────────────────────────────────────────────────
    println!("\nPhase 3 ▸ Summary");
    println!("  Classical A*   — global optimum via admissible Manhattan heuristic");
    println!("  Quantum QUBO   — one-hot route selection via Simulated Annealing");
    println!("  Crates used    — pathfinding 4.15, ndarray 0.17, quantrs2-tytan 0.2");
}

// ─────────────────────────────────────────────────────────────────────────────
// Terrain

/// Build a {}³ terrain: mostly cost-1 cells with two high-cost obstacle slabs.
fn build_terrain() -> Array3<u32> {
    let mut costs = vec![1u32; GRID * GRID * GRID];

    let idx = |x: usize, y: usize, z: usize| x * GRID * GRID + y * GRID + z;

    // Slab 1: wall at x = GRID/3, single gap at y = GRID/2
    for y in 0..GRID {
        for z in 0..GRID {
            if y != GRID / 2 {
                costs[idx(GRID / 3, y, z)] = 50;
            }
        }
    }

    // Slab 2: wall at z = 2*GRID/3, single gap at x = GRID/2
    for x in 0..GRID {
        for y in 0..GRID {
            if x != GRID / 2 {
                costs[idx(x, y, 2 * GRID / 3)] = 50;
            }
        }
    }

    Array3::from_shape_vec((GRID, GRID, GRID), costs).unwrap()
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1: A*

/// Run A* from `start` to `goal` on the 3D cost grid.
/// Edge cost = cost of the destination cell.
fn find_path_astar(grid: &Array3<u32>, start: Pos, goal: Pos) -> (Vec<Pos>, u32) {
    let g = GRID as i32;
    let (path, cost) = astar(
        &start,
        |&(x, y, z)| {
            [
                (1i32, 0i32, 0i32),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ]
            .iter()
            .filter_map(|&(dx, dy, dz)| {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                let nz = z as i32 + dz;
                if (0..g).contains(&nx) && (0..g).contains(&ny) && (0..g).contains(&nz) {
                    let p = (nx as usize, ny as usize, nz as usize);
                    Some((p, grid[[p.0, p.1, p.2]]))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
        },
        |&(x, y, z)| {
            (goal.0.abs_diff(x) + goal.1.abs_diff(y) + goal.2.abs_diff(z)) as u32
        },
        |&p| p == goal,
    )
    .expect("A* found no path — check grid connectivity");
    (path, cost)
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2: path enumeration + QUBO

/// DFS-based enumeration of simple paths from `from` to `to`.
///
/// `max_hops`  – depth limit (paths longer than this are discarded)
/// `max_count` – stop after collecting this many paths
///
/// Returns `(path, total_grid_cost)` where cost excludes the starting node.
pub fn enumerate_paths(
    grid: &Array3<u32>,
    from: Pos,
    to: Pos,
    max_hops: usize,
    max_count: usize,
) -> Vec<(Vec<Pos>, u32)> {
    let mut results = Vec::new();
    let mut visited = vec![from];
    dfs_paths(grid, from, to, max_hops, &mut visited, 0, &mut results, max_count);
    results
}

fn dfs_paths(
    grid: &Array3<u32>,
    pos: Pos,
    goal: Pos,
    hops_left: usize,
    path: &mut Vec<Pos>,
    cost: u32,
    results: &mut Vec<(Vec<Pos>, u32)>,
    max_count: usize,
) {
    if results.len() >= max_count {
        return;
    }
    if pos == goal {
        results.push((path.clone(), cost));
        return;
    }
    if hops_left == 0 {
        return;
    }
    let g = GRID as i32;
    let (x, y, z) = pos;
    for (dx, dy, dz) in [
        (1i32, 0i32, 0i32),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ] {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        let nz = z as i32 + dz;
        if (0..g).contains(&nx) && (0..g).contains(&ny) && (0..g).contains(&nz) {
            let next = (nx as usize, ny as usize, nz as usize);
            if !path.contains(&next) {
                let step_cost = grid[[next.0, next.1, next.2]];
                path.push(next);
                dfs_paths(grid, next, goal, hops_left - 1, path, cost + step_cost, results, max_count);
                path.pop();
            }
        }
    }
}

/// Build a one-hot QUBO that selects the minimum-cost route from `alternatives`.
///
/// Hamiltonian:
///   H = Σᵢ costᵢ · xᵢ  +  λ · (Σᵢ xᵢ – 1)²
///
/// Expanding and mapping to the symmetric matrix E = Σᵢⱼ Q[i,j]·xᵢ·xⱼ:
///   Q[i,i] = costᵢ – λ
///   Q[i,j] = λ   for i ≠ j   (symmetric; combined contribution = 2λ per pair)
fn build_one_hot_qubo(
    alternatives: &[(Vec<Pos>, u32)],
) -> (Array2<f64>, HashMap<String, usize>) {
    let n = alternatives.len();
    let max_cost = alternatives.iter().map(|(_, c)| *c).max().unwrap_or(1) as f64;
    // λ must exceed the maximum possible cost difference to enforce one-hot
    let lambda = max_cost * 2.0 + 1.0;

    let mut q = Array2::<f64>::zeros((n, n));
    let mut var_map = HashMap::new();

    for (i, (_, cost)) in alternatives.iter().enumerate() {
        var_map.insert(format!("r{i}"), i);
        q[[i, i]] = *cost as f64 - lambda;
        for j in (i + 1)..n {
            // Symmetric: each pair (i,j) contributes λ + λ = 2λ to energy
            q[[i, j]] = lambda;
            q[[j, i]] = lambda;
        }
    }

    (q, var_map)
}
