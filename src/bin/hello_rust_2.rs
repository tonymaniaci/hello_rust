//! Quantum-Classical Pathfinder v2 — BFS baseline + QUBO savings demo
//!
//! Difference from v1: BFS is cost-blind (finds min-hop path only).
//! High-cost obstacles are injected onto that BFS path so the QUBO
//! has real savings to find by routing around them.

use ndarray::{Array2, Array3};
use pathfinding::prelude::bfs;
use quantrs2_tytan::sampler::{TabuSampler, Sampler};
use std::collections::HashMap;

type Pos = (usize, usize, usize);

const GRID: usize = 20;
const WINDOW: usize = 5;
const MAX_ALTS: usize = 12;
const OBSTACLE_COST: u32 = 18; // High-cost cells injected onto the BFS route

// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Quantum-Classical Pathfinder v2 — BFS + QUBO   ║");
    println!("╚══════════════════════════════════════════════════╝\n");

    let start: Pos = (0, 0, 0);
    let goal: Pos = (GRID - 1, GRID - 1, GRID - 1);

    // ── Step 1: BFS on uniform terrain (all cells cost 1) ───────────────────
    println!("Step 1 ▸ BFS on uniform terrain (cost-blind, min-hop only)");

    let mut grid = Array3::<u32>::ones((GRID, GRID, GRID));
    let bfs_path = find_path_bfs(start, goal);

    println!("  BFS path length : {} nodes", bfs_path.len());

    // ── Step 2: Inject expensive obstacles onto the BFS path ────────────────
    println!("\nStep 2 ▸ Injecting high-cost obstacles onto the BFS path");

    let injection_indices = [8usize, 20, 32, 44];
    for &idx in &injection_indices {
        if idx < bfs_path.len() {
            let (x, y, z) = bfs_path[idx];
            grid[[x, y, z]] = OBSTACLE_COST;
            println!("  obstacle at path[{:>2}] = {:?}  cost → {}", idx, bfs_path[idx], OBSTACLE_COST);
        }
    }

    let bfs_cost: u32 = bfs_path[1..].iter().map(|&p| grid[[p.0, p.1, p.2]]).sum();
    println!("\n  BFS path cost on obstacle terrain : {}", bfs_cost);
    println!("  (BFS is blind to costs — it walked straight into the obstacles)");

    // ── Step 3: QUBO refinement window around each obstacle ─────────────────
    println!("\nStep 3 ▸ QUBO route refinement (SASampler, quantrs2-tytan)");

    let mut total_savings: u32 = 0;
    let mut refined_cost = bfs_cost;

    for &inject_idx in &injection_indices {
        let window_start = inject_idx.saturating_sub(2);
        let window_end = (window_start + WINDOW).min(bfs_path.len() - 1);

        if window_start >= window_end || inject_idx >= bfs_path.len() {
            continue;
        }

        let from = bfs_path[window_start];
        let to = bfs_path[window_end];

        // Enumerate alternative routes through this window
        let mut alts = enumerate_paths(&grid, from, to, WINDOW + 2, MAX_ALTS - 1);

        // Include the BFS sub-segment as baseline (r0)
        let bfs_sub: Vec<Pos> = bfs_path[window_start..=window_end].to_vec();
        let bfs_sub_cost: u32 = bfs_sub[1..].iter().map(|&p| grid[[p.0, p.1, p.2]]).sum();
        alts.insert(0, (bfs_sub, bfs_sub_cost));
        alts.dedup_by_key(|(_, c)| *c);

        println!(
            "\n  Window path[{}..{}]  obstacle at [{}]  BFS sub-cost: {}",
            window_start, window_end, inject_idx, bfs_sub_cost
        );

        if alts.len() < 2 {
            println!("  No alternatives found — keeping BFS route.");
            continue;
        }

        println!("  {} routes found:", alts.len());
        for (i, (_, c)) in alts.iter().enumerate() {
            let tag = if i == 0 { " ← BFS baseline" } else { "" };
            println!("    r{i}: cost {c}{tag}");
        }

        let (qubo_matrix, var_map) = build_one_hot_qubo(&alts);
        let sampler = TabuSampler::new();
        let results = sampler
            .run_qubo(&(qubo_matrix, var_map), 300)
            .expect("TabuSampler failed");

        let best = &results[0];
        let selected_idx = (0..alts.len())
            .find(|&i| *best.assignments.get(&format!("r{i}")).unwrap_or(&false))
            .unwrap_or(0);

        let (_, selected_cost) = &alts[selected_idx];
        println!("  QUBO energy: {:.3}  →  selected r{selected_idx} (cost {selected_cost})", best.energy);

        if *selected_cost < bfs_sub_cost {
            let saving = bfs_sub_cost - selected_cost;
            total_savings += saving;
            refined_cost -= saving;
            println!(
                "  ✓ Savings: {} units ({:.1}% cheaper for this segment)",
                saving,
                100.0 * saving as f64 / bfs_sub_cost as f64
            );
        } else {
            println!("  No improvement for this window.");
        }
    }

    // ── Final Summary ────────────────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║                  Final Summary                   ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!("  Initial BFS cost    : {}", bfs_cost);
    println!("  QUBO refined cost   : {}", refined_cost);
    println!("  Total savings       : {}", total_savings);
    println!(
        "  Overall improvement : {:.1}%",
        if bfs_cost > 0 { 100.0 * total_savings as f64 / bfs_cost as f64 } else { 0.0 }
    );
    println!("\n  BFS  — cost-blind, walks straight into obstacles");
    println!("  QUBO — quantum-inspired, routes around obstacles via SA solver");
    println!("  Crates — pathfinding 4.15, ndarray 0.17, quantrs2-tytan 0.2");
}

// ─────────────────────────────────────────────────────────────────────────────
// BFS: minimum hops, ignores cell costs entirely

fn find_path_bfs(start: Pos, goal: Pos) -> Vec<Pos> {
    let g = GRID as i32;
    bfs(
        &start,
        |&(x, y, z)| {
            [(1i32, 0i32, 0i32), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
                .iter()
                .filter_map(|&(dx, dy, dz)| {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    if (0..g).contains(&nx) && (0..g).contains(&ny) && (0..g).contains(&nz) {
                        Some((nx as usize, ny as usize, nz as usize))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        },
        |&p| p == goal,
    )
    .expect("BFS found no path")
}

// ─────────────────────────────────────────────────────────────────────────────
// DFS-based path enumeration (same logic as v1)

fn enumerate_paths(
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
        (1i32, 0i32, 0i32), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    ] {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        let nz = z as i32 + dz;
        if (0..g).contains(&nx) && (0..g).contains(&ny) && (0..g).contains(&nz) {
            let next = (nx as usize, ny as usize, nz as usize);
            if !path.contains(&next) {
                path.push(next);
                dfs_paths(
                    grid, next, goal, hops_left - 1, path,
                    cost + grid[[next.0, next.1, next.2]], results, max_count,
                );
                path.pop();
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// One-hot QUBO: select the minimum-cost route from a list of alternatives
//
//   H = Σᵢ costᵢ·xᵢ  +  λ·(Σᵢ xᵢ – 1)²
//
//   Q[i,i] = costᵢ – λ
//   Q[i,j] = λ  for i ≠ j  (symmetric; full energy contribution = 2λ per pair)

fn build_one_hot_qubo(
    alternatives: &[(Vec<Pos>, u32)],
) -> (Array2<f64>, HashMap<String, usize>) {
    let n = alternatives.len();
    let max_cost = alternatives.iter().map(|(_, c)| *c).max().unwrap_or(1) as f64;
    let lambda = max_cost * 2.0 + 1.0;

    let mut q = Array2::<f64>::zeros((n, n));
    let mut var_map = HashMap::new();

    for (i, (_, cost)) in alternatives.iter().enumerate() {
        var_map.insert(format!("r{i}"), i);
        q[[i, i]] = *cost as f64 - lambda;
        for j in (i + 1)..n {
            q[[i, j]] = lambda; // upper-triangular only; TabuSampler uses energy = Σᵢⱼ Q[i,j]·xᵢ·xⱼ
        }
    }

    (q, var_map)
}
