// Case-driven host for the Metal kernels of the Swift FDTD programs (interim mode of docs/FDTD_BACKEND_SPEC.md):
// LoopFDTD/dipole_s11.swift (6 fields) and MEEPhistopheles/src/metal/dipole_s11.swift (Ez, Hx, Hy).
// The kernels are not in this file: metal_interim.py extracts `metalCode` verbatim from the program (optionally
// with a recorded patch) into kernel.swift and compiles both together. Both programs bind
//   updateH(F[0..<nF], Db, Nx, Ny, Nz)   and   updateE(F[0..<nF], isPEC, Cb, Nx, Ny, Nz)
// with the E fields first, so one host serves both from n_fields / n_e in config.json.
// Per step it does what the programs' own hosts do (one encoder: updateH then updateE, wait, then the soft sources
// and probe reads on the CPU). Usage: host IN_DIR (reads config.json, pec.u8, src_idx.i32, src_amp.f32,
// probe_idx.i32; writes probes.f32, final_E.f32, timing.json).
import Foundation
import Metal

let dir = URL(fileURLWithPath: CommandLine.arguments[1])

func load<T>(_ name: String, _: T.Type) -> [T] {
    let d = try! Data(contentsOf: dir.appendingPathComponent(name))
    return d.withUnsafeBytes { Array($0.bindMemory(to: T.self)) }
}

func save<T>(_ name: String, _ a: [T]) {
    a.withUnsafeBytes { try! Data($0).write(to: dir.appendingPathComponent(name)) }
}

let cfg = try! JSONSerialization.jsonObject(
    with: Data(contentsOf: dir.appendingPathComponent("config.json"))) as! [String: Any]
let dims = cfg["N"] as! [Int]
let Nx = dims[0], Ny = dims[1], Nz = dims[2]
let steps = cfg["steps"] as! Int
let nF = cfg["n_fields"] as! Int
let nE = cfg["n_e"] as! Int
let cbBits = UInt32(cfg["Cb_bits"] as! Int)          // float32 bit patterns: no decimal round trip
let dbBits = UInt32(cfg["Db_bits"] as! Int)
var Cb = Float(bitPattern: cbBits)
var Db = Float(bitPattern: dbBits)
let gridSize = Nx * Ny * Nz

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    print("no Metal device"); exit(1)
}
let bytes = gridSize * MemoryLayout<Float>.stride
let F: [MTLBuffer] = (0..<nF).map { _ in device.makeBuffer(length: bytes, options: .storageModeShared)! }
for b in F { memset(b.contents(), 0, bytes) }
let pec = load("pec.u8", UInt8.self)
precondition(pec.count == gridSize, "pec.u8 has \(pec.count) cells, grid has \(gridSize)")
let pecBuf = device.makeBuffer(bytes: pec, length: gridSize, options: .storageModeShared)!

let library = try! device.makeLibrary(source: metalCode, options: nil)
let pipeE = try! device.makeComputePipelineState(function: library.makeFunction(name: "updateE")!)
let pipeH = try! device.makeComputePipelineState(function: library.makeFunction(name: "updateH")!)

var nx = UInt32(Nx), ny = UInt32(Ny), nz = UInt32(Nz)
let srcIdx = load("src_idx.i32", Int32.self)          // (field buffer, flat index, waveform row) triples
let srcAmp = load("src_amp.f32", Float.self)          // waveform rows x steps
let prIdx = load("probe_idx.i32", Int32.self)         // (field buffer, flat index) pairs
let nsrc = srcIdx.count / 3, npr = prIdx.count / 2
let ptr = F.map { $0.contents().bindMemory(to: Float.self, capacity: gridSize) }
var out = [Float](repeating: 0, count: steps * npr)

let grid = MTLSize(width: Nx, height: Ny, depth: Nz)
let group = MTLSize(width: 4, height: 4, depth: 4)    // as in the program
let t0 = Date()
for n in 0..<steps {
    let cmd = queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeH)
    for i in 0..<nF { enc.setBuffer(F[i], offset: 0, index: i) }
    enc.setBytes(&Db, length: 4, index: nF)
    enc.setBytes(&nx, length: 4, index: nF + 1); enc.setBytes(&ny, length: 4, index: nF + 2); enc.setBytes(&nz, length: 4, index: nF + 3)
    enc.dispatchThreads(grid, threadsPerThreadgroup: group)
    enc.setComputePipelineState(pipeE)
    for i in 0..<nF { enc.setBuffer(F[i], offset: 0, index: i) }
    enc.setBuffer(pecBuf, offset: 0, index: nF)
    enc.setBytes(&Cb, length: 4, index: nF + 1)
    enc.setBytes(&nx, length: 4, index: nF + 2); enc.setBytes(&ny, length: 4, index: nF + 3); enc.setBytes(&nz, length: 4, index: nF + 4)
    enc.dispatchThreads(grid, threadsPerThreadgroup: group)
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    for s in 0..<nsrc {
        ptr[Int(srcIdx[3 * s])][Int(srcIdx[3 * s + 1])] += srcAmp[Int(srcIdx[3 * s + 2]) * steps + n]
    }
    for p in 0..<npr {
        out[n * npr + p] = ptr[Int(prIdx[2 * p])][Int(prIdx[2 * p + 1])]
    }
}
let wall = Date().timeIntervalSince(t0)

save("probes.f32", out)
var finalE = Data()
for c in 0..<nE { finalE.append(Data(bytes: F[c].contents(), count: bytes)) }
try! finalE.write(to: dir.appendingPathComponent("final_E.f32"))
try! JSONSerialization.data(withJSONObject: ["wall_s": wall, "device": device.name])
    .write(to: dir.appendingPathComponent("timing.json"))
print("host: \(steps) steps in \(String(format: "%.2f", wall)) s on \(device.name)")
