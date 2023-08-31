/***************************************************************************************
* Copyright (c) 2020-2021 Institute of Computing Technology, Chinese Academy of Sciences
* Copyright (c) 2020-2021 Peng Cheng Laboratory
*
* XiangShan is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
*
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
*
* See the Mulan PSL v2 for more details.
***************************************************************************************/

package xiangshan.backend

import chipsalliance.rocketchip.config.Parameters
import chisel3._
import chisel3.util._
import freechips.rocketchip.diplomacy.{BundleBridgeSource, LazyModule, LazyModuleImp}
import freechips.rocketchip.tile.HasFPUParameters
import freechips.rocketchip.tilelink.TLBuffer
import coupledL2.PrefetchRecv
import utils._
import utility._
import xiangshan._
import xiangshan.backend.exu.StdExeUnit
import xiangshan.backend.fu._
import xiangshan.backend.rob.{DebugLSIO, LsTopdownInfo, RobLsqIO, RobPtr}
import xiangshan.cache._
import xiangshan.cache.mmu._
import xiangshan.mem._
import xiangshan.mem.mdp._
import xiangshan.mem.prefetch.{BasePrefecher, SMSParams, SMSPrefetcher}

class Std(implicit p: Parameters) extends FunctionUnit {
  io.in.ready := true.B
  io.out.valid := io.in.valid
  io.out.bits.uop := io.in.bits.uop
  io.out.bits.data := io.in.bits.src(0)
}

class ooo_to_mem(implicit p: Parameters) extends XSBundle{
  val loadFastMatch = Vec(exuParameters.LduCnt, Input(UInt(exuParameters.LduCnt.W)))
  val loadFastImm = Vec(exuParameters.LduCnt, Input(UInt(12.W)))
  val sfence = Input(new SfenceBundle)
  val tlbCsr = Input(new TlbCsrBundle)
  val lsqio = new Bundle {
   val lcommit = Input(UInt(log2Up(CommitWidth + 1).W))
   val scommit = Input(UInt(log2Up(CommitWidth + 1).W))
   val pendingld = Input(Bool())
   val pendingst = Input(Bool())
   val commit = Input(Bool())
   val pendingPtr = Input(new RobPtr)
  }

  val isStore = Input(Bool())
  val csrCtrl = Flipped(new CustomCSRCtrlIO)
  val enqLsq = new LsqEnqIO
  val flushSb = Input(Bool())
  val loadPc = Vec(exuParameters.LduCnt, Input(UInt(VAddrBits.W))) // for hw prefetch
  val issue = Vec(exuParameters.LsExuCnt + exuParameters.StuCnt, Flipped(DecoupledIO(new ExuInput)))
}

class mem_to_ooo(implicit p: Parameters ) extends XSBundle{
  val otherFastWakeup = Vec(exuParameters.LduCnt + 2 * exuParameters.StuCnt, ValidIO(new MicroOp))
  val csrUpdate = new DistributedCSRUpdateReq
  val lqCancelCnt = Output(UInt(log2Up(VirtualLoadQueueSize + 1).W))
  val sqCancelCnt = Output(UInt(log2Up(StoreQueueSize + 1).W))
  val sqDeq = Output(UInt(log2Ceil(EnsbufferWidth + 1).W))
  val lqDeq = Output(UInt(log2Up(CommitWidth + 1).W))
  val stIn = Vec(exuParameters.StuCnt, ValidIO(new ExuInput))
  val stIssuePtr = Output(new SqPtr())

  val memoryViolation = ValidIO(new Redirect)
  val sbIsEmpty = Output(Bool())

  val lsTopdownInfo = Vec(exuParameters.LduCnt, Output(new LsTopdownInfo))

  val lsqio = new Bundle {
    val vaddr = Output(UInt(VAddrBits.W))
    val mmio = Output(Vec(LoadPipelineWidth, Bool()))
    val uop = Output(Vec(LoadPipelineWidth, new MicroOp))
    val lqCanAccept = Output(Bool())
    val sqCanAccept = Output(Bool())
  }
  val writeback = Vec(exuParameters.LsExuCnt + exuParameters.StuCnt, DecoupledIO(new ExuOutput))
}

class fetch_to_mem(implicit p: Parameters) extends XSBundle{
  val itlb = Flipped(new TlbPtwIO())
}


class MemBlock()(implicit p: Parameters) extends LazyModule
  with HasXSParameter with HasWritebackSource {

  val dcache = LazyModule(new DCacheWrapper())
  val uncache = LazyModule(new Uncache())
  val ptw = LazyModule(new L2TLBWrapper())
  val ptw_to_l2_buffer = if (!coreParams.softPTW) LazyModule(new TLBuffer) else null
  val pf_sender_opt = coreParams.prefetcher.map(_ =>
    BundleBridgeSource(() => new PrefetchRecv)
  )

  if (!coreParams.softPTW) {
    ptw_to_l2_buffer.node := ptw.node
  }

  lazy val module = new MemBlockImp(this)

  override val writebackSourceParams: Seq[WritebackSourceParams] = {
    val params = new WritebackSourceParams
    params.exuConfigs = (loadExuConfigs ++ storeExuConfigs).map(cfg => Seq(cfg))
    Seq(params)
  }
  override lazy val writebackSourceImp: HasWritebackSourceImp = module
}

class MemBlockImp(outer: MemBlock) extends LazyModuleImp(outer)
  with HasXSParameter
  with HasFPUParameters
  with HasWritebackSourceImp
  with HasPerfEvents
{

  val io = IO(new Bundle {
    val hartId = Input(UInt(8.W))
    val redirect = Flipped(ValidIO(new Redirect))
<<<<<<< Updated upstream
=======
<<<<<<< HEAD
    // in
    val issue = Vec(exuParameters.LsExuCnt + exuParameters.StuCnt, Flipped(DecoupledIO(new ExuInput)))
    val loadFastMatch = Vec(exuParameters.LduCnt, Input(UInt(exuParameters.LduCnt.W)))
    val loadFastImm = Vec(exuParameters.LduCnt, Input(UInt(12.W)))
    // TODO: 谁给谁的信号?
=======
>>>>>>> Stashed changes

    val ooo_to_mem = new ooo_to_mem
    val mem_to_ooo = new mem_to_ooo
    val fetch_to_mem = new fetch_to_mem

<<<<<<< Updated upstream
=======
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes
    val rsfeedback = Vec(exuParameters.LsExuCnt, new MemRSFeedbackIO)


    val int2vlsu = Flipped(new Int2VLSUIO)
    val vec2vlsu = Flipped(new Vec2VLSUIO)
    // out
    val s3_delayed_load_error = Vec(exuParameters.LduCnt, Output(Bool()))
    val vlsu2vec = new VLSU2VecIO
    val vlsu2int = new VLSU2IntIO
    val vlsu2ctrl = new VLSU2CtrlIO
    // prefetch to l1 req
    val prefetch_req = Flipped(DecoupledIO(new L1PrefetchReq))
    // misc
    val error = new L1CacheErrorInfo
    val memInfo = new Bundle {
      val sqFull = Output(Bool())
      val lqFull = Output(Bool())
      val dcacheMSHRFull = Output(Bool())
    }
    val debug_ls = new DebugLSIO
    val lsTopdownInfo = Vec(exuParameters.LduCnt, Output(new LsTopdownInfo))
    val l2_hint = Input(Valid(new L2ToL1Hint()))
  })

  override def writebackSource1: Option[Seq[Seq[DecoupledIO[ExuOutput]]]] = Some(Seq(io.mem_to_ooo.writeback))

  val redirect = RegNextWithEnable(io.redirect)

  val dcache = outer.dcache.module
  val uncache = outer.uncache.module

  val delayedDcacheRefill = RegNext(dcache.io.lsu.lsq)

<<<<<<< Updated upstream
  val csrCtrl = DelayN(io.ooo_to_mem.csrCtrl, 2)
  dcache.io.csr.distribute_csr <> csrCtrl.distribute_csr
  dcache.io.l2_pf_store_only := RegNext(io.ooo_to_mem.csrCtrl.l2_pf_store_only, false.B)
  io.mem_to_ooo.csrUpdate := RegNext(dcache.io.csr.update)
=======
<<<<<<< HEAD
  //TODO: 为什么delay2? 分布式的csrCtrl是不是需要2个cycle才能拿到?
  val csrCtrl = DelayN(io.csrCtrl, 2)
  // 把延迟2cycle之后的信号送到dcache
  dcache.io.csr.distribute_csr <> csrCtrl.distribute_csr
  // 延迟一拍给dcache
  // TODO: 这里为什么延迟数有区别?
  dcache.io.l2_pf_store_only := RegNext(io.csrCtrl.l2_pf_store_only, false.B)
  // dcache中, 把更新csr的请求送出去
  io.csrUpdate := RegNext(dcache.io.csr.update)
  // dcache中, 把dcache发生error更新csr的请求送出去
=======
  val csrCtrl = DelayN(io.ooo_to_mem.csrCtrl, 2)
  dcache.io.csr.distribute_csr <> csrCtrl.distribute_csr
  dcache.io.l2_pf_store_only := RegNext(io.ooo_to_mem.csrCtrl.l2_pf_store_only, false.B)
  io.mem_to_ooo.csrUpdate := RegNext(dcache.io.csr.update)
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes
  io.error <> RegNext(RegNext(dcache.io.error))
  when(!csrCtrl.cache_error_enable){
    io.error.report_to_beu := false.B
    io.error.valid := false.B
  }

  val loadUnits = Seq.fill(exuParameters.LduCnt)(Module(new LoadUnit))
  val storeUnits = Seq.fill(exuParameters.StuCnt)(Module(new StoreUnit))
  val stdExeUnits = Seq.fill(exuParameters.StuCnt)(Module(new StdExeUnit))
  val stData = stdExeUnits.map(_.io.out)
  val exeUnits = loadUnits ++ storeUnits
  val l1_pf_req = Wire(Decoupled(new L1PrefetchReq()))
  val prefetcherOpt: Option[BasePrefecher] = coreParams.prefetcher.map {
    case _: SMSParams =>
      val sms = Module(new SMSPrefetcher())
<<<<<<< Updated upstream
      sms.io_agt_en := RegNextN(io.ooo_to_mem.csrCtrl.l1D_pf_enable_agt, 2, Some(false.B))
      sms.io_pht_en := RegNextN(io.ooo_to_mem.csrCtrl.l1D_pf_enable_pht, 2, Some(false.B))
      sms.io_act_threshold := RegNextN(io.ooo_to_mem.csrCtrl.l1D_pf_active_threshold, 2, Some(12.U))
      sms.io_act_stride := RegNextN(io.ooo_to_mem.csrCtrl.l1D_pf_active_stride, 2, Some(30.U))
      sms.io_stride_en := RegNextN(io.ooo_to_mem.csrCtrl.l1D_pf_enable_stride, 2, Some(true.B))
      sms
  }
  prefetcherOpt.foreach(pf => {
    val pf_to_l2 = ValidIODelay(pf.io.l2_req, 2)
=======
<<<<<<< HEAD
      // 把配置SMS的信息通过csr写入SMS中
      sms.io_agt_en := RegNextN(io.csrCtrl.l1D_pf_enable_agt, 2, Some(false.B))
      sms.io_pht_en := RegNextN(io.csrCtrl.l1D_pf_enable_pht, 2, Some(false.B))
      sms.io_act_threshold := RegNextN(io.csrCtrl.l1D_pf_active_threshold, 2, Some(12.U))
      sms.io_act_stride := RegNextN(io.csrCtrl.l1D_pf_active_stride, 2, Some(30.U))
      sms.io_stride_en := RegNextN(io.csrCtrl.l1D_pf_enable_stride, 2, Some(true.B))
      sms
  }
  prefetcherOpt.foreach(pf => {
    // pf_addr是从sms预取器中发出的预取地址
    val pf_to_l2 = ValidIODelay(pf.io.pf_addr, 2)
    // 把预取地址送到MemBlock的pf_sender_opt中, 进而把该地址送到L2中
=======
      sms.io_agt_en := RegNextN(io.ooo_to_mem.csrCtrl.l1D_pf_enable_agt, 2, Some(false.B))
      sms.io_pht_en := RegNextN(io.ooo_to_mem.csrCtrl.l1D_pf_enable_pht, 2, Some(false.B))
      sms.io_act_threshold := RegNextN(io.ooo_to_mem.csrCtrl.l1D_pf_active_threshold, 2, Some(12.U))
      sms.io_act_stride := RegNextN(io.ooo_to_mem.csrCtrl.l1D_pf_active_stride, 2, Some(30.U))
      sms.io_stride_en := RegNextN(io.ooo_to_mem.csrCtrl.l1D_pf_enable_stride, 2, Some(true.B))
      sms
  }
  prefetcherOpt.foreach(pf => {
    val pf_to_l2 = ValidIODelay(pf.io.l2_req, 2)
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes
    outer.pf_sender_opt.get.out.head._1.addr_valid := pf_to_l2.valid
    outer.pf_sender_opt.get.out.head._1.addr := pf_to_l2.bits.addr
    outer.pf_sender_opt.get.out.head._1.pf_source := pf_to_l2.bits.source
    outer.pf_sender_opt.get.out.head._1.l2_pf_en := RegNextN(io.ooo_to_mem.csrCtrl.l2_pf_enable, 2, Some(true.B))
    pf.io.enable := RegNextN(io.ooo_to_mem.csrCtrl.l1D_pf_enable, 2, Some(false.B))
  })
  prefetcherOpt match {
    case Some(pf) => l1_pf_req <> pf.io.l1_req
    case None =>
      l1_pf_req.valid := false.B
      l1_pf_req.bits := DontCare
  }
<<<<<<< Updated upstream
  val pf_train_on_hit = RegNextN(io.ooo_to_mem.csrCtrl.l1D_pf_train_on_hit, 2, Some(true.B))
=======
<<<<<<< HEAD
  //默认使能预取器在发生hit时继续训练
  val pf_train_on_hit = RegNextN(io.csrCtrl.l1D_pf_train_on_hit, 2, Some(true.B))
=======
  val pf_train_on_hit = RegNextN(io.ooo_to_mem.csrCtrl.l1D_pf_train_on_hit, 2, Some(true.B))
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes

  loadUnits.zipWithIndex.map(x => x._1.suggestName("LoadUnit_"+x._2))
  storeUnits.zipWithIndex.map(x => x._1.suggestName("StoreUnit_"+x._2))
  val atomicsUnit = Module(new AtomicsUnit)

  // Atom inst comes from sta / std, then its result
  // will be writebacked using load writeback port
  //
  // However, atom exception will be writebacked to rob
  // using store writeback port

<<<<<<< Updated upstream

=======
<<<<<<< HEAD
>>>>>>> Stashed changes
  // TODO: atomic写回端口占用了loadUnits.head的写回端口, 那么lodUnits.head的写回怎么办? 直接丢弃会不会出错?
  // atomic指令会清空rob才dispatch，因此不会出现上述情况

  val loadWritebackOverride  = Mux(atomicsUnit.io.out.valid, atomicsUnit.io.out.bits, loadUnits.head.io.loadOut.bits)
  val loadOut0 = Wire(Decoupled(new ExuOutput))
  loadOut0.valid := atomicsUnit.io.out.valid || loadUnits.head.io.loadOut.valid
  loadOut0.bits  := loadWritebackOverride
  atomicsUnit.io.out.ready := loadOut0.ready
  loadUnits.head.io.loadOut.ready := loadOut0.ready

  // 如果写回的是atomicsUnit, 则atmoicsUnit的异常信号是从store写回, 而不是从loadUnits写回
  // 所以这里把loadUnits的异常向量全部清空
<<<<<<< Updated upstream
  // TODO: 为什么atomic的异常是从store写回？ 因为atomic指令(例如lr)失败与否与store指令(例如sc)的执行结果相关， 与load的执行结果无关。

=======
  // TODO: 为什么atomic的异常是从store写回？ 因为atomic指令(例如lr)失败与否与store指令(例如sr)的执行结果相关， 与load的执行结果无关。
=======
  val loadWritebackOverride  = Mux(atomicsUnit.io.out.valid, atomicsUnit.io.out.bits, loadUnits.head.io.ldout.bits)
  val ldout0 = Wire(Decoupled(new ExuOutput))
  ldout0.valid := atomicsUnit.io.out.valid || loadUnits.head.io.ldout.valid
  ldout0.bits  := loadWritebackOverride
  atomicsUnit.io.out.ready := ldout0.ready
  loadUnits.head.io.ldout.ready := ldout0.ready
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes
  when(atomicsUnit.io.out.valid){
    ldout0.bits.uop.cf.exceptionVec := 0.U(16.W).asBools // exception will be writebacked via store wb port
  }

<<<<<<< Updated upstream
=======
<<<<<<< HEAD
  // 这里取tail, 是因为loadUnits第一个写回是由atomic和loadUnits_0共享的, 也就是loadOut0
  val ldExeWbReqs = loadOut0 +: loadUnits.tail.map(_.io.loadOut)
  io.writeback <> ldExeWbReqs ++ VecInit(storeUnits.map(_.io.stout)) ++ VecInit(stdExeUnits.map(_.io.out))
  io.otherFastWakeup := DontCare
  // TODO: 这里是硬编码, 获取loadUnits, 应该改为take(exuParameters.LduCnt)
  // 把从loadUnits中送出的fastUop信号通过MemBlock送到顶层的XSCore中
  // XSCore会把这些信号再送回exuBlocks中唤醒相应指令继续执行
  io.otherFastWakeup.take(2).zip(loadUnits.map(_.io.fastUop)).foreach{case(a,b)=> a := b}
  // 丢掉load和std的写回, 只剩sta的写回, 用于后续trigger和mmio的判断
  val stOut = io.writeback.drop(exuParameters.LduCnt).dropRight(exuParameters.StuCnt)
=======
>>>>>>> Stashed changes
  val ldExeWbReqs = ldout0 +: loadUnits.tail.map(_.io.ldout)
  io.mem_to_ooo.writeback <> ldExeWbReqs ++ VecInit(storeUnits.map(_.io.stout)) ++ VecInit(stdExeUnits.map(_.io.out))
  io.mem_to_ooo.otherFastWakeup := DontCare
  io.mem_to_ooo.otherFastWakeup.take(2).zip(loadUnits.map(_.io.fast_uop)).foreach{case(a,b)=> a := b}
  val stOut = io.mem_to_ooo.writeback.drop(exuParameters.LduCnt).dropRight(exuParameters.StuCnt)
<<<<<<< Updated upstream
=======
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes

  // prefetch to l1 req
  loadUnits.foreach(load_unit => {
    load_unit.io.prefetch_req.valid <> l1_pf_req.valid
    load_unit.io.prefetch_req.bits <> l1_pf_req.bits
  })
  // when loadUnits(0) stage 0 is busy, hw prefetch will never use that pipeline
  loadUnits(0).io.prefetch_req.bits.confidence := 0.U

  l1_pf_req.ready := (l1_pf_req.bits.confidence > 0.U) ||
    loadUnits.map(!_.io.ldin.valid).reduce(_ || _)

  // l1 pf fuzzer interface
  val DebugEnableL1PFFuzzer = false
  if (DebugEnableL1PFFuzzer) {
    // l1 pf req fuzzer
    val fuzzer = Module(new L1PrefetchFuzzer())
    fuzzer.io.vaddr := DontCare
    fuzzer.io.paddr := DontCare

    // override load_unit prefetch_req
    loadUnits.foreach(load_unit => {
      load_unit.io.prefetch_req.valid <> fuzzer.io.req.valid
      load_unit.io.prefetch_req.bits <> fuzzer.io.req.bits
    })

    fuzzer.io.req.ready := l1_pf_req.ready
  }

  // TODO: fast load wakeup
  val lsq     = Module(new LsqWrapper)
  val vlsq    = Module(new DummyVectorLsq)
  val sbuffer = Module(new Sbuffer)
  // if you wants to stress test dcache store, use FakeSbuffer
  // val sbuffer = Module(new FakeSbuffer) // out of date now
<<<<<<< Updated upstream
  io.mem_to_ooo.stIssuePtr := lsq.io.issuePtrExt
=======
<<<<<<< HEAD
  // store queue中issue指针, 输出给MemBlock, 最终送到执行单元
  // 如果配置了checkWait则只有当其他指令在这个store后面时, 才允许发射
  io.stIssuePtr := lsq.io.issuePtrExt
=======
  io.mem_to_ooo.stIssuePtr := lsq.io.issuePtrExt
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes

  dcache.io.hartId := io.hartId
  lsq.io.hartId := io.hartId
  sbuffer.io.hartId := io.hartId
  atomicsUnit.io.hartId := io.hartId

  // ptw
  val sfence = RegNext(RegNext(io.ooo_to_mem.sfence))
  val tlbcsr = RegNext(RegNext(io.ooo_to_mem.tlbCsr))
  val ptw = outer.ptw.module
  val ptw_to_l2_buffer = outer.ptw_to_l2_buffer.module
  ptw.io.sfence <> sfence
  ptw.io.csr.tlb <> tlbcsr
  ptw.io.csr.distribute_csr <> csrCtrl.distribute_csr
  ptw.io.tlb(0) <> io.fetch_to_mem.itlb

  val perfEventsPTW = Wire(Vec(19, new PerfEvent))
  if (!coreParams.softPTW) {
    perfEventsPTW := ptw.getPerf
  } else {
    perfEventsPTW := DontCare
  }

  // dtlb
<<<<<<< Updated upstream
=======
<<<<<<< HEAD
  // TODO: 为什么打两拍?
  // 从外部输入的sfence
  val sfence = RegNext(RegNext(io.sfence))
  // 从csr寄存器传来的信息
  val tlbcsr = RegNext(RegNext(io.tlbCsr))
  // 为load store prefetch单独建立tlb
=======
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes
  val dtlb_ld = VecInit(Seq.fill(1){
    val tlb_ld = Module(new TLBNonBlock(exuParameters.LduCnt, 2, ldtlbParams))
    tlb_ld.io // let the module have name in waveform
  })
  val dtlb_st = VecInit(Seq.fill(1){
    val tlb_st = Module(new TLBNonBlock(exuParameters.StuCnt, 1, sttlbParams))
    tlb_st.io // let the module have name in waveform
  })
  val dtlb_prefetch = VecInit(Seq.fill(1){
    val tlb_prefetch = Module(new TLBNonBlock(1, 2, pftlbParams))
    tlb_prefetch.io // let the module have name in waveform
  })
  val dtlb = dtlb_ld ++ dtlb_st ++ dtlb_prefetch
<<<<<<< Updated upstream
  val ptwio = Wire(new VectorTlbPtwIO(exuParameters.LduCnt + exuParameters.StuCnt + 1)) // load + store + hw prefetch
=======
<<<<<<< HEAD
  // 展平所有dtlb的requestor, 生成一个新的Seq
  // requestor中信号包含req/req_kill/response
=======
  val ptwio = Wire(new VectorTlbPtwIO(exuParameters.LduCnt + exuParameters.StuCnt + 1)) // load + store + hw prefetch
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes
  val dtlb_reqs = dtlb.map(_.requestor).flatten
  val dtlb_pmps = dtlb.map(_.pmp).flatten
  dtlb.map(_.sfence := sfence)
  dtlb.map(_.csr := tlbcsr)
  dtlb.map(_.flushPipe.map(a => a := false.B)) // non-block doesn't need
  if (refillBothTlb) {
    require(ldtlbParams.outReplace == sttlbParams.outReplace)
    require(ldtlbParams.outReplace)

    val replace = Module(new TlbReplace(exuParameters.LduCnt + exuParameters.StuCnt + 1, ldtlbParams))
    replace.io.apply_sep(dtlb_ld.map(_.replace) ++ dtlb_st.map(_.replace), ptwio.resp.bits.data.entry.tag)
  } else {
    if (ldtlbParams.outReplace) {
      val replace_ld = Module(new TlbReplace(exuParameters.LduCnt, ldtlbParams))
      replace_ld.io.apply_sep(dtlb_ld.map(_.replace), ptwio.resp.bits.data.entry.tag)
    }
    if (sttlbParams.outReplace) {
      val replace_st = Module(new TlbReplace(exuParameters.StuCnt, sttlbParams))
      replace_st.io.apply_sep(dtlb_st.map(_.replace), ptwio.resp.bits.data.entry.tag)
    }
  }

<<<<<<< Updated upstream
  val ptw_resp_next = RegEnable(ptwio.resp.bits, ptwio.resp.valid)
  val ptw_resp_v = RegNext(ptwio.resp.valid && !(sfence.valid && tlbcsr.satp.changed), init = false.B)
  ptwio.resp.ready := true.B
=======
<<<<<<< HEAD
  val ptw_resp_next = RegEnable(io.ptw.resp.bits, io.ptw.resp.valid)
  // 只有在sfence没执行或satp没有改变情况下ptw的response才有效
  // The SFENCE.VMA is used to flush any local hardware caches related to address translation.
  val ptw_resp_v = RegNext(io.ptw.resp.valid && !(sfence.valid && tlbcsr.satp.changed), init = false.B)
  io.ptw.resp.ready := true.B
=======
  val ptw_resp_next = RegEnable(ptwio.resp.bits, ptwio.resp.valid)
  val ptw_resp_v = RegNext(ptwio.resp.valid && !(sfence.valid && tlbcsr.satp.changed), init = false.B)
  ptwio.resp.ready := true.B
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes

  dtlb.flatMap(a => a.ptw.req)
    .zipWithIndex
    .foreach{ case (tlb, i) =>
<<<<<<< Updated upstream
      tlb.ready := ptwio.req(i).ready
      ptwio.req(i).bits := tlb.bits
    val vector_hit = if (refillBothTlb) Cat(ptw_resp_next.vector).orR
      else if (i < exuParameters.LduCnt) Cat(ptw_resp_next.vector.take(exuParameters.LduCnt)).orR
      else Cat(ptw_resp_next.vector.drop(exuParameters.LduCnt)).orR
    ptwio.req(i).valid := tlb.valid && !(ptw_resp_v && vector_hit &&
=======
<<<<<<< HEAD
    // 把dtlb里面的ptw.req和MemBlock的io.ptw.req连起来
    tlb <> io.ptw.req(i)
    val vector_hit = if (refillBothTlb) Cat(ptw_resp_next.vector).orR
      else if (i < exuParameters.LduCnt) Cat(ptw_resp_next.vector.take(exuParameters.LduCnt)).orR
      else Cat(ptw_resp_next.vector.drop(exuParameters.LduCnt)).orR
      // 是否需要真的发出ptw请求? 只有在dtlb中ptw.req有效, 且满足如下条件时才发出:
      // ptw_resp_v为假或者vector_hit为假或者ptw_resp_next没有hit
      // 也就是当拍没有ptw的response, 且当拍不会回填ptw时才会发出ptw请求
    io.ptw.req(i).valid := tlb.valid && !(ptw_resp_v && vector_hit &&
=======
      tlb.ready := ptwio.req(i).ready
      ptwio.req(i).bits := tlb.bits
    val vector_hit = if (refillBothTlb) Cat(ptw_resp_next.vector).orR
      else if (i < exuParameters.LduCnt) Cat(ptw_resp_next.vector.take(exuParameters.LduCnt)).orR
      else Cat(ptw_resp_next.vector.drop(exuParameters.LduCnt)).orR
    ptwio.req(i).valid := tlb.valid && !(ptw_resp_v && vector_hit &&
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes
      ptw_resp_next.data.hit(tlb.bits.vpn, tlbcsr.satp.asid, allType = true, ignoreAsid = true))
  }
  dtlb.foreach(_.ptw.resp.bits := ptw_resp_next.data)
  if (refillBothTlb) {
    dtlb.foreach(_.ptw.resp.valid := ptw_resp_v && Cat(ptw_resp_next.vector).orR)
  } else {
    dtlb_ld.foreach(_.ptw.resp.valid := ptw_resp_v && Cat(ptw_resp_next.vector.take(exuParameters.LduCnt)).orR)
    dtlb_st.foreach(_.ptw.resp.valid := ptw_resp_v && Cat(ptw_resp_next.vector.drop(exuParameters.LduCnt).take(exuParameters.StuCnt)).orR)
    dtlb_prefetch.foreach(_.ptw.resp.valid := ptw_resp_v && Cat(ptw_resp_next.vector.drop(exuParameters.LduCnt + exuParameters.StuCnt)).orR)
  }

  val dtlbRepeater1  = PTWFilter(ldtlbParams.fenceDelay, ptwio, sfence, tlbcsr, l2tlbParams.dfilterSize)
  val dtlbRepeater2  = PTWRepeaterNB(passReady = false, ldtlbParams.fenceDelay, dtlbRepeater1.io.ptw, ptw.io.tlb(1), sfence, tlbcsr)
  val itlbRepeater2 = PTWRepeaterNB(passReady = false, itlbParams.fenceDelay, io.fetch_to_mem.itlb, ptw.io.tlb(0), sfence, tlbcsr)
<<<<<<< Updated upstream

  ExcitingUtils.addSource(dtlbRepeater1.io.rob_head_miss_in_tlb, s"miss_in_dtlb_${coreParams.HartId}", ExcitingUtils.Perf, true)

  // pmp
  val pmp = Module(new PMP())
  pmp.io.distribute_csr <> csrCtrl.distribute_csr

  val pmp_check = VecInit(Seq.fill(exuParameters.LduCnt + exuParameters.StuCnt + 1)(Module(new PMPChecker(3)).io))
  for ((p,d) <- pmp_check zip dtlb_pmps) {
    p.apply(tlbcsr.priv.dmode, pmp.io.pmp, pmp.io.pma, d)
    require(p.req.bits.size.getWidth == d.bits.size.getWidth)
  }
  for (i <- 0 until 8) {
    val pmp_check_ptw = Module(new PMPCheckerv2(lgMaxSize = 3, sameCycle = false, leaveHitMux = true))
    pmp_check_ptw.io.apply(tlbcsr.priv.dmode, pmp.io.pmp, pmp.io.pma, ptwio.resp.valid,
      Cat(ptwio.resp.bits.data.entry.ppn, ptwio.resp.bits.data.ppn_low(i), 0.U(12.W)).asUInt)
    dtlb.map(_.ptw_replenish(i) := pmp_check_ptw.io.resp)
  }

  for (i <- 0 until exuParameters.LduCnt) {
    io.debug_ls.debugLsInfo(i) := loadUnits(i).io.debug_ls
  }
  for (i <- 0 until exuParameters.StuCnt) {
    io.debug_ls.debugLsInfo(i + exuParameters.LduCnt) := storeUnits(i).io.debug_ls
  }

  io.mem_to_ooo.lsTopdownInfo := loadUnits.map(_.io.lsTopdownInfo)

=======

  ExcitingUtils.addSource(dtlbRepeater1.io.rob_head_miss_in_tlb, s"miss_in_dtlb_${coreParams.HartId}", ExcitingUtils.Perf, true)

  // pmp
  val pmp = Module(new PMP())
  // csr是分布式的, 当指令更新csr时, 需要同步这些所有分布式的csr
  pmp.io.distribute_csr <> csrCtrl.distribute_csr

  // 针对每个load/store/prefetcher unit都单独有一个PMPChecker
  // 把dtlb的pmp请求连接到PMPChecker
  val pmp_check = VecInit(Seq.fill(exuParameters.LduCnt + exuParameters.StuCnt + 1)(Module(new PMPChecker(3)).io))
  for ((p,d) <- pmp_check zip dtlb_pmps) {
    p.apply(tlbcsr.priv.dmode, pmp.io.pmp, pmp.io.pma, d)
    require(p.req.bits.size.getWidth == d.bits.size.getWidth)
  }
  // TODO: tlbcontiguous =8, 这段代码的作用? 对于连续的页面翻译
  for (i <- 0 until 8) {
    val pmp_check_ptw = Module(new PMPCheckerv2(lgMaxSize = 3, sameCycle = false, leaveHitMux = true))
    pmp_check_ptw.io.apply(tlbcsr.priv.dmode, pmp.io.pmp, pmp.io.pma, ptwio.resp.valid,
      Cat(ptwio.resp.bits.data.entry.ppn, ptwio.resp.bits.data.ppn_low(i), 0.U(12.W)).asUInt)
    dtlb.map(_.ptw_replenish(i) := pmp_check_ptw.io.resp)
  }

<<<<<<< HEAD
  // TODO: 这部分是调试用的trigger, 具体trigger是否enbale由csr控制
  // 如果使能trigger, 则把相应的trigger数据写入loadUnits或者storeUnits的相应地址
=======
  for (i <- 0 until exuParameters.LduCnt) {
    io.debug_ls.debugLsInfo(i) := loadUnits(i).io.debug_ls
  }
  for (i <- 0 until exuParameters.StuCnt) {
    io.debug_ls.debugLsInfo(i + exuParameters.LduCnt) := storeUnits(i).io.debug_ls
  }

  io.mem_to_ooo.lsTopdownInfo := loadUnits.map(_.io.lsTopdownInfo)

>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes
  val tdata = RegInit(VecInit(Seq.fill(6)(0.U.asTypeOf(new MatchTriggerIO))))
  val tEnable = RegInit(VecInit(Seq.fill(6)(false.B)))
  val en = csrCtrl.trigger_enable
  tEnable := VecInit(en(2), en (3), en(4), en(5), en(7), en(9))
  when(csrCtrl.mem_trigger.t.valid) {
    tdata(csrCtrl.mem_trigger.t.bits.addr) := csrCtrl.mem_trigger.t.bits.tdata
  }
  val lTriggerMapping = Map(0 -> 2, 1 -> 3, 2 -> 5)
  val sTriggerMapping = Map(0 -> 0, 1 -> 1, 2 -> 4)
  val lChainMapping = Map(0 -> 2)
  val sChainMapping = Map(0 -> 1)
  XSDebug(tEnable.asUInt.orR, "Debug Mode: At least one store trigger is enabled\n")
  for(j <- 0 until 3)
    PrintTriggerInfo(tEnable(j), tdata(j))

  // LoadUnit
  class BalanceEntry extends XSBundle {
    val balance = Bool()
    val req = new LqWriteBundle
    val port = UInt(log2Up(LoadPipelineWidth).W)
  }

  def balanceReOrder(sel: Seq[ValidIO[BalanceEntry]]): Seq[ValidIO[BalanceEntry]] = {
    require(sel.length > 0)
    val balancePick = ParallelPriorityMux(sel.map(x => (x.valid && x.bits.balance) -> x))
    val reorderSel = Wire(Vec(sel.length, ValidIO(new BalanceEntry)))
    (0 until sel.length).map(i =>
      if (i == 0) {
        when (balancePick.valid && balancePick.bits.balance) {
          reorderSel(i) := balancePick
        } .otherwise {
          reorderSel(i) := sel(i)
        }
      } else {
        when (balancePick.valid && balancePick.bits.balance && i.U === balancePick.bits.port) {
          reorderSel(i) := sel(0)
        } .otherwise {
          reorderSel(i) := sel(i)
        }
      }
    )
    reorderSel
  }

  val fastReplaySel = loadUnits.zipWithIndex.map { case (ldu, i) => {
    val wrapper = Wire(Valid(new BalanceEntry))
    wrapper.valid        := ldu.io.fast_rep_out.valid
    wrapper.bits.req     := ldu.io.fast_rep_out.bits
    wrapper.bits.balance := ldu.io.fast_rep_out.bits.rep_info.bank_conflict
    wrapper.bits.port    := i.U
    wrapper
  }}
  val balanceFastReplaySel = balanceReOrder(fastReplaySel)

  for (i <- 0 until exuParameters.LduCnt) {
    loadUnits(i).io.redirect <> redirect
    loadUnits(i).io.isFirstIssue := true.B
<<<<<<< Updated upstream
=======
<<<<<<< HEAD
  
    // get input form dispatch
    loadUnits(i).io.loadIn <> io.issue(i)
    // 从load pipe传回replay信号给RS
    loadUnits(i).io.feedbackSlow <> io.rsfeedback(i).feedbackSlow
    loadUnits(i).io.feedbackFast <> io.rsfeedback(i).feedbackFast
    loadUnits(i).io.rsIdx := io.rsfeedback(i).rsIdx
   
    // fast replay
    loadUnits(i).io.fastReplayIn.valid := balanceFastReplaySel(i).valid 
    loadUnits(i).io.fastReplayIn.bits := balanceFastReplaySel(i).bits.req
=======
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes

    // get input form dispatch
    loadUnits(i).io.ldin <> io.ooo_to_mem.issue(i)
    loadUnits(i).io.feedback_slow <> io.rsfeedback(i).feedbackSlow
    loadUnits(i).io.feedback_fast <> io.rsfeedback(i).feedbackFast
    loadUnits(i).io.rsIdx := io.rsfeedback(i).rsIdx

    // fast replay
    loadUnits(i).io.fast_rep_in.valid := balanceFastReplaySel(i).valid
    loadUnits(i).io.fast_rep_in.bits := balanceFastReplaySel(i).bits.req

    loadUnits(i).io.fast_rep_out.ready := false.B
    for (j <- 0 until exuParameters.LduCnt) {
      when (balanceFastReplaySel(j).valid && balanceFastReplaySel(j).bits.port === i.U) {
        loadUnits(i).io.fast_rep_out.ready := loadUnits(j).io.fast_rep_in.ready
      }
    }

    // get input form dispatch
<<<<<<< Updated upstream
    loadUnits(i).io.ldin <> io.ooo_to_mem.issue(i)
=======
<<<<<<< HEAD
    // TODO: 前面也有这句话, 冗余?
    loadUnits(i).io.loadIn <> io.issue(i)
=======
    loadUnits(i).io.ldin <> io.ooo_to_mem.issue(i)
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes
    // dcache access
    loadUnits(i).io.dcache <> dcache.io.lsu.load(i)
    // forward
    loadUnits(i).io.lsq.forward <> lsq.io.forward(i)
    loadUnits(i).io.sbuffer <> sbuffer.io.forward(i)
<<<<<<< Updated upstream
    loadUnits(i).io.tl_d_channel := dcache.io.lsu.forward_D(i)
    loadUnits(i).io.forward_mshr <> dcache.io.lsu.forward_mshr(i)
    // ld-ld violation check
    loadUnits(i).io.lsq.ldld_nuke_query <> lsq.io.ldu.ldld_nuke_query(i)
    loadUnits(i).io.lsq.stld_nuke_query <> lsq.io.ldu.stld_nuke_query(i)
=======
<<<<<<< HEAD
    // 在发出Tilink的命令是TLMessages.GrantData时, dcache可以forward给loadUnits数据
    loadUnits(i).io.tlDchannel := dcache.io.lsu.forward_D(i)
    loadUnits(i).io.forward_mshr <> dcache.io.lsu.forward_mshr(i)
    // ld-ld violation check
    // load的S2会去check是否发生了violation, 把信号送到lsq去确认
    loadUnits(i).io.lsq.loadLoadViolationQuery <> lsq.io.ldu.loadLoadViolationQuery(i)
    loadUnits(i).io.lsq.storeLoadViolationQuery <> lsq.io.ldu.storeLoadViolationQuery(i)
=======
    loadUnits(i).io.tl_d_channel := dcache.io.lsu.forward_D(i)
    loadUnits(i).io.forward_mshr <> dcache.io.lsu.forward_mshr(i)
    // ld-ld violation check
    loadUnits(i).io.lsq.ldld_nuke_query <> lsq.io.ldu.ldld_nuke_query(i)
    loadUnits(i).io.lsq.stld_nuke_query <> lsq.io.ldu.stld_nuke_query(i)
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes
    loadUnits(i).io.csrCtrl       <> csrCtrl
    // dcache refill req
    loadUnits(i).io.refill           <> delayedDcacheRefill
    // dtlb
    loadUnits(i).io.tlb <> dtlb_reqs.take(exuParameters.LduCnt)(i)
    // pmp
    // 把loadUnits中访问pmp的请求送到pmp去检查，返回resp
    loadUnits(i).io.pmp <> pmp_check(i).resp
    // st-ld violation query
    for (s <- 0 until StorePipelineWidth) {
<<<<<<< Updated upstream
      //store S1输出给ldu, 看看有些load是否需要重新执行
      loadUnits(i).io.reExecuteQuery(s) := storeUnits(s).io.reExecuteQuery
    }
    loadUnits(i).io.lq_rep_full <> lsq.io.lq_rep_full
=======
<<<<<<< HEAD
      // store流水线中执行的指令, 送到loadUnits中进行检查, 是否发生violation
      loadUnits(i).io.reExecuteQuery(s) := storeUnits(s).io.reExecuteQuery
    }
    // 如果lsq满了, 反馈信号给loadUnits, 用于触发fastReplay
    loadUnits(i).io.lqReplayFull <> lsq.io.lqReplayFull
=======
      loadUnits(i).io.stld_nuke_query(s) := storeUnits(s).io.stld_nuke_query
    }
    loadUnits(i).io.lq_rep_full <> lsq.io.lq_rep_full
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes
    // prefetch
    prefetcherOpt.foreach(pf => {
      pf.io.ld_in(i).valid := Mux(pf_train_on_hit,
        loadUnits(i).io.prefetch_train.valid,
        loadUnits(i).io.prefetch_train.valid && loadUnits(i).io.prefetch_train.bits.isFirstIssue && (
          loadUnits(i).io.prefetch_train.bits.miss || loadUnits(i).io.prefetch_train.bits.meta_prefetch
          )
      )
      pf.io.ld_in(i).bits := loadUnits(i).io.prefetch_train.bits
<<<<<<< Updated upstream
      pf.io.ld_in(i).bits.uop.cf.pc := Mux(loadUnits(i).io.s2_ptr_chasing, io.ooo_to_mem.loadPc(i), RegNext(io.ooo_to_mem.loadPc(i)))
=======
<<<<<<< HEAD
      // 如果是pointerChasing则可以把pc直接送回, 否则要打一拍保证时序
      pf.io.ld_in(i).bits.uop.cf.pc := Mux(loadUnits(i).io.s2IsPointerChasing, io.loadPc(i), RegNext(io.loadPc(i)))
=======
      pf.io.ld_in(i).bits.uop.cf.pc := Mux(loadUnits(i).io.s2_ptr_chasing, io.ooo_to_mem.loadPc(i), RegNext(io.ooo_to_mem.loadPc(i)))
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes
    })

    // load to load fast forward: load(i) prefers data(i)
    val fastPriority = (i until exuParameters.LduCnt) ++ (0 until i)
<<<<<<< Updated upstream
    val fastValidVec = fastPriority.map(j => loadUnits(j).io.l2l_fwd_out.valid)
    val fastDataVec = fastPriority.map(j => loadUnits(j).io.l2l_fwd_out.data)
    val fastErrorVec = fastPriority.map(j => loadUnits(j).io.l2l_fwd_out.dly_ld_err)
    val fastMatchVec = fastPriority.map(j => io.ooo_to_mem.loadFastMatch(i)(j))
    loadUnits(i).io.l2l_fwd_in.valid := VecInit(fastValidVec).asUInt.orR
    loadUnits(i).io.l2l_fwd_in.data := ParallelPriorityMux(fastValidVec, fastDataVec)
    loadUnits(i).io.l2l_fwd_in.dly_ld_err := ParallelPriorityMux(fastValidVec, fastErrorVec)
=======
<<<<<<< HEAD
    // 结合上面, loadUnit0产生Seq(0, 1), fastValidVec就是(loadUnits(0).io.fastpathOut.valid, loadUnits(1).io.fastpathOut.valid)
    // loadUnit(1)的fastValidVec就是(loadUnits(1).io.fastpathOut.valid, loadUnits(0).io.fastpathOut.valid)
    // 假如: loadUnits(0).io.fastpathOut.valid = true.B;  loadUnits(1).io.fastpathOut.valid = false.B
    // fastValidVec = Seq(true.B, false.B);
    val fastValidVec = fastPriority.map(j => loadUnits(j).io.fastpathOut.valid)
    val fastDataVec = fastPriority.map(j => loadUnits(j).io.fastpathOut.data)
    // 对于loadUnit0: fastMatchVec = (io.loadFastMatch(0)(0), io.loadFastMatch(0)(1))
    // 对于loadUnit1: fastMatchVec = (io.loadFastMatch(1)(1), io.loadFastMatch(1)(0))
    val fastMatchVec = fastPriority.map(j => io.loadFastMatch(i)(j))
    loadUnits(i).io.fastpathIn.valid := VecInit(fastValidVec).asUInt.orR
    loadUnits(i).io.fastpathIn.data := ParallelPriorityMux(fastValidVec, fastDataVec)
>>>>>>> Stashed changes
    val fastMatch = ParallelPriorityMux(fastValidVec, fastMatchVec)
    loadUnits(i).io.ld_fast_match := fastMatch
    loadUnits(i).io.ld_fast_imm := io.ooo_to_mem.loadFastImm(i)
    loadUnits(i).io.replay <> lsq.io.replay(i)

<<<<<<< Updated upstream
    loadUnits(i).io.l2_hint <> io.l2_hint
=======
    // TODO: 这个hint作用是什么? L2用来告诉loadUnits,是否要尽快发起replay
    // 最终信号来源是CustomL1Hint.scala中的l1Hint, 如果L1说3个cycle后就能获取数据, 那么loadUnits就尽快发起fastReplay
    loadUnits(i).io.l2Hint <> io.l2Hint
=======
    val fastValidVec = fastPriority.map(j => loadUnits(j).io.l2l_fwd_out.valid)
    val fastDataVec = fastPriority.map(j => loadUnits(j).io.l2l_fwd_out.data)
    val fastErrorVec = fastPriority.map(j => loadUnits(j).io.l2l_fwd_out.dly_ld_err)
    val fastMatchVec = fastPriority.map(j => io.ooo_to_mem.loadFastMatch(i)(j))
    loadUnits(i).io.l2l_fwd_in.valid := VecInit(fastValidVec).asUInt.orR
    loadUnits(i).io.l2l_fwd_in.data := ParallelPriorityMux(fastValidVec, fastDataVec)
    loadUnits(i).io.l2l_fwd_in.dly_ld_err := ParallelPriorityMux(fastValidVec, fastErrorVec)
    val fastMatch = ParallelPriorityMux(fastValidVec, fastMatchVec)
    loadUnits(i).io.ld_fast_match := fastMatch
    loadUnits(i).io.ld_fast_imm := io.ooo_to_mem.loadFastImm(i)
    loadUnits(i).io.replay <> lsq.io.replay(i)

    loadUnits(i).io.l2_hint <> io.l2_hint
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes

    // passdown to lsq (load s2)
<<<<<<< Updated upstream
    lsq.io.ldu.ldin(i) <> loadUnits(i).io.lsq.ldin
    lsq.io.ldout(i) <> loadUnits(i).io.lsq.uncache
    lsq.io.ld_raw_data(i) <> loadUnits(i).io.lsq.ld_raw_data
    lsq.io.trigger(i) <> loadUnits(i).io.lsq.trigger

    lsq.io.l2_hint.valid := io.l2_hint.valid
    lsq.io.l2_hint.bits.sourceId := io.l2_hint.bits.sourceId
=======
<<<<<<< HEAD
    // 从loadUnits写回数据到load Queue
    lsq.io.ldu.loadIn(i) <> loadUnits(i).io.lsq.loadIn
    // TODO: loadRawDataOut和loadOut中的data什么区别?
    lsq.io.loadOut(i) <> loadUnits(i).io.lsq.loadOut
    // loadUnits从loadQueue中拿Raw数据
    lsq.io.ldRawDataOut(i) <> loadUnits(i).io.lsq.ldRawData
    // 把loadUnits和lsq的trigger连起来
    lsq.io.trigger(i) <> loadUnits(i).io.lsq.trigger

    // 把l2Hint送给lsq
    lsq.io.l2Hint.valid := io.l2Hint.valid
    lsq.io.l2Hint.bits.sourceId := io.l2Hint.bits.sourceId
=======
    lsq.io.ldu.ldin(i) <> loadUnits(i).io.lsq.ldin
    lsq.io.ldout(i) <> loadUnits(i).io.lsq.uncache
    lsq.io.ld_raw_data(i) <> loadUnits(i).io.lsq.ld_raw_data
    lsq.io.trigger(i) <> loadUnits(i).io.lsq.trigger

    lsq.io.l2_hint.valid := io.l2_hint.valid
    lsq.io.l2_hint.bits.sourceId := io.l2_hint.bits.sourceId
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes

    // alter writeback exception info
    io.s3_delayed_load_error(i) := loadUnits(i).io.s3_dly_ld_err

    // update mem dependency predictor
    // io.memPredUpdate(i) := DontCare

    // --------------------------------
    // Load Triggers
    // --------------------------------
    val hit = Wire(Vec(3, Bool()))
    for (j <- 0 until 3) {
      loadUnits(i).io.trigger(j).tdata2 := tdata(lTriggerMapping(j)).tdata2
      loadUnits(i).io.trigger(j).matchType := tdata(lTriggerMapping(j)).matchType
      loadUnits(i).io.trigger(j).tEnable := tEnable(lTriggerMapping(j))
      // Just let load triggers that match data unavailable
      hit(j) := loadUnits(i).io.trigger(j).addrHit && !tdata(lTriggerMapping(j)).select // Mux(tdata(j + 3).select, loadUnits(i).io.trigger(j).lastDataHit, loadUnits(i).io.trigger(j).addrHit)
      io.mem_to_ooo.writeback(i).bits.uop.cf.trigger.backendHit(lTriggerMapping(j)) := hit(j)
//      io.writeback(i).bits.uop.cf.trigger.backendTiming(lTriggerMapping(j)) := tdata(lTriggerMapping(j)).timing
      //      if (lChainMapping.contains(j)) io.writeback(i).bits.uop.cf.trigger.triggerChainVec(lChainMapping(j)) := hit && tdata(j+3).chain
    }
    when(tdata(2).chain) {
      io.mem_to_ooo.writeback(i).bits.uop.cf.trigger.backendHit(2) := hit(0) && hit(1)
      io.mem_to_ooo.writeback(i).bits.uop.cf.trigger.backendHit(3) := hit(0) && hit(1)
    }
    when(!io.mem_to_ooo.writeback(i).bits.uop.cf.trigger.backendEn(1)) {
      io.mem_to_ooo.writeback(i).bits.uop.cf.trigger.backendHit(5) := false.B
    }

    XSDebug(io.mem_to_ooo.writeback(i).bits.uop.cf.trigger.getHitBackend && io.mem_to_ooo.writeback(i).valid, p"Debug Mode: Load Inst No.${i}" +
    p"has trigger hit vec ${io.mem_to_ooo.writeback(i).bits.uop.cf.trigger.backendHit}\n")

  }
  // Prefetcher
  val PrefetcherDTLBPortIndex = exuParameters.LduCnt + exuParameters.StuCnt
  prefetcherOpt match {
  case Some(pf) => dtlb_reqs(PrefetcherDTLBPortIndex) <> pf.io.tlb_req
  case None =>
    dtlb_reqs(PrefetcherDTLBPortIndex) := DontCare
    dtlb_reqs(PrefetcherDTLBPortIndex).req.valid := false.B
    dtlb_reqs(PrefetcherDTLBPortIndex).resp.ready := true.B
  }

  // 连接StoreUnit
  for (i <- 0 until exuParameters.StuCnt) {
    val stu = storeUnits(i)

    // 连接store data 流水线
    stdExeUnits(i).io.redirect <> redirect
<<<<<<< Updated upstream
    stdExeUnits(i).io.fromInt <> io.ooo_to_mem.issue(i + exuParameters.LduCnt + exuParameters.StuCnt)
=======
<<<<<<< HEAD
    // issue = LsExuCnt + StuCnt, 其中LsExuCnt = LduCnt + StuCnt
    // 展开后issue = LduCnt + StuCnt + StuCnt, 这里的i就是用来取出store data的
    stdExeUnits(i).io.fromInt <> io.issue(i + exuParameters.LduCnt + exuParameters.StuCnt)
>>>>>>> Stashed changes
    stdExeUnits(i).io.fromFp := DontCare
    stdExeUnits(i).io.out := DontCare

    stu.io.redirect     <> redirect
    // 从stu输出到rsfeedback, 由于rsfeedback是和ldu一起计算, 因此前面加上LduCnt后表示是stu的开始
    stu.io.feedbackSlow <> io.rsfeedback(exuParameters.LduCnt + i).feedbackSlow
    // 从rs输出的rsIdx
    stu.io.rsIdx        <> io.rsfeedback(exuParameters.LduCnt + i).rsIdx
=======
    stdExeUnits(i).io.fromInt <> io.ooo_to_mem.issue(i + exuParameters.LduCnt + exuParameters.StuCnt)
    stdExeUnits(i).io.fromFp := DontCare
    stdExeUnits(i).io.out := DontCare

    stu.io.redirect      <> redirect
    stu.io.feedback_slow <> io.rsfeedback(exuParameters.LduCnt + i).feedbackSlow
    stu.io.rsIdx         <> io.rsfeedback(exuParameters.LduCnt + i).rsIdx
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
    // NOTE: just for dtlb's perf cnt
    // 从rs输出的isFirstIssue
    stu.io.isFirstIssue <> io.rsfeedback(exuParameters.LduCnt + i).isFirstIssue
<<<<<<< Updated upstream
    // 从rs输入的uop
    stu.io.stin         <> io.issue(exuParameters.LduCnt + i)
    // 从stu的s1 stage输出到lsq
=======
    stu.io.stin         <> io.ooo_to_mem.issue(exuParameters.LduCnt + i)
>>>>>>> Stashed changes
    stu.io.lsq          <> lsq.io.sta.storeAddrIn(i)
    // 从stu的s2 stage输出到lsq
    stu.io.lsq_replenish <> lsq.io.sta.storeAddrInRe(i)
    // dtlb
    // stu的tlb请求和response
    stu.io.tlb          <> dtlb_reqs.drop(exuParameters.LduCnt)(i)
    // 从pmp返回的check结果给stu
    stu.io.pmp          <> pmp_check(i+exuParameters.LduCnt).resp

    // store unit does not need fast feedback
    // storeunit没有快速唤醒通路
    io.rsfeedback(exuParameters.LduCnt + i).feedbackFast := DontCare

    // Lsq to sta unit
<<<<<<< Updated upstream
    // stu在S0输出给lsq的store mask
    lsq.io.sta.storeMaskIn(i) <> stu.io.storeMaskOut
=======
    lsq.io.sta.storeMaskIn(i) <> stu.io.st_mask_out
>>>>>>> Stashed changes

    // Lsq to std unit's rs
    // store data输出给lsq
    lsq.io.std.storeDataIn(i) := stData(i)


    // 1. sync issue info to store set LFST
    // 2. when store issue, broadcast issued sqPtr to wake up the following insts
    // io.stIn(i).valid := io.issue(exuParameters.LduCnt + i).valid
    // io.stIn(i).bits := io.issue(exuParameters.LduCnt + i).bits
<<<<<<< Updated upstream
    io.mem_to_ooo.stIn(i).valid := stu.io.issue.valid
    io.mem_to_ooo.stIn(i).bits := stu.io.issue.bits
=======
<<<<<<< HEAD
    // stu中指令发射后, 通过stIn发送到LFST去更新表中内容
    io.stIn(i).valid := stu.io.issue.valid
    io.stIn(i).bits := stu.io.issue.bits
=======
    io.mem_to_ooo.stIn(i).valid := stu.io.issue.valid
    io.mem_to_ooo.stIn(i).bits := stu.io.issue.bits
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes

    stu.io.stout.ready := true.B

    // -------------------------
    // Store Triggers
    // -------------------------
    when(stOut(i).fire()){
      val hit = Wire(Vec(3, Bool()))
      for (j <- 0 until 3) {
         hit(j) := !tdata(sTriggerMapping(j)).select && TriggerCmp(
           stOut(i).bits.debug.vaddr,
           tdata(sTriggerMapping(j)).tdata2,
           tdata(sTriggerMapping(j)).matchType,
           tEnable(sTriggerMapping(j))
         )
       stOut(i).bits.uop.cf.trigger.backendHit(sTriggerMapping(j)) := hit(j)
     }

     when(tdata(0).chain) {
       io.mem_to_ooo.writeback(i).bits.uop.cf.trigger.backendHit(0) := hit(0) && hit(1)
       io.mem_to_ooo.writeback(i).bits.uop.cf.trigger.backendHit(1) := hit(0) && hit(1)
     }

     when(!stOut(i).bits.uop.cf.trigger.backendEn(0)) {
       stOut(i).bits.uop.cf.trigger.backendHit(4) := false.B
     }
   }
  }

  // mmio store writeback will use store writeback port 0
  lsq.io.mmioStout.ready := false.B
  when (lsq.io.mmioStout.valid && !storeUnits(0).io.stout.valid) {
    stOut(0).valid := true.B
    stOut(0).bits  := lsq.io.mmioStout.bits
    lsq.io.mmioStout.ready := true.B
  }

  // atomic exception / trigger writeback
  when (atomicsUnit.io.out.valid) {
    // atom inst will use store writeback port 0 to writeback exception info
    stOut(0).valid := true.B
    stOut(0).bits  := atomicsUnit.io.out.bits
    assert(!lsq.io.mmioStout.valid && !storeUnits(0).io.stout.valid)

    // when atom inst writeback, surpress normal load trigger
    (0 until exuParameters.LduCnt).map(i => {
      io.mem_to_ooo.writeback(i).bits.uop.cf.trigger.backendHit := VecInit(Seq.fill(6)(false.B))
    })
  }

  // Uncahce
<<<<<<< Updated upstream
  uncache.io.enableOutstanding := io.ooo_to_mem.csrCtrl.uncache_write_outstanding_enable
=======
<<<<<<< HEAD
  // 目前uncache的写不支持outstanding
  uncache.io.enableOutstanding := io.csrCtrl.uncache_write_outstanding_enable
=======
  uncache.io.enableOutstanding := io.ooo_to_mem.csrCtrl.uncache_write_outstanding_enable
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes
  uncache.io.hartId := io.hartId
  lsq.io.uncacheOutstanding := io.ooo_to_mem.csrCtrl.uncache_write_outstanding_enable

  // Lsq
  io.mem_to_ooo.lsqio.mmio       := lsq.io.rob.mmio
  io.mem_to_ooo.lsqio.uop        := lsq.io.rob.uop
  lsq.io.rob.lcommit             := io.ooo_to_mem.lsqio.lcommit
  lsq.io.rob.scommit             := io.ooo_to_mem.lsqio.scommit
  lsq.io.rob.pendingld           := io.ooo_to_mem.lsqio.pendingld
  lsq.io.rob.pendingst           := io.ooo_to_mem.lsqio.pendingst
  lsq.io.rob.commit              := io.ooo_to_mem.lsqio.commit
  lsq.io.rob.pendingPtr          := io.ooo_to_mem.lsqio.pendingPtr

//  lsq.io.rob            <> io.lsqio.rob
  lsq.io.enq            <> io.ooo_to_mem.enqLsq
  lsq.io.brqRedirect    <> redirect
<<<<<<< Updated upstream
  io.mem_to_ooo.memoryViolation    <> lsq.io.rollback
  io.mem_to_ooo.lsqio.lqCanAccept  := lsq.io.lqCanAccept
  io.mem_to_ooo.lsqio.sqCanAccept  := lsq.io.sqCanAccept
=======
<<<<<<< HEAD
  // 把从loadQueue中拿到的是否发生memoryViolation信号送出去(ctrlBlock)处理
  // ctrlBlock会基于该信号从redirectGen模块中产生flush信号
  io.memoryViolation    <> lsq.io.rollback
  io.lsqio.lqCanAccept  := lsq.io.lqCanAccept
  io.lsqio.sqCanAccept  := lsq.io.sqCanAccept
=======
  io.mem_to_ooo.memoryViolation    <> lsq.io.rollback
  io.mem_to_ooo.lsqio.lqCanAccept  := lsq.io.lqCanAccept
  io.mem_to_ooo.lsqio.sqCanAccept  := lsq.io.sqCanAccept
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes
  // lsq.io.uncache        <> uncache.io.lsq
  AddPipelineReg(lsq.io.uncache.req, uncache.io.lsq.req, false.B)
  AddPipelineReg(uncache.io.lsq.resp, lsq.io.uncache.resp, false.B)
  // delay dcache refill for 1 cycle for better timing
  lsq.io.refill         := delayedDcacheRefill
  lsq.io.release        := dcache.io.lsu.release
<<<<<<< Updated upstream
=======
<<<<<<< HEAD
  lsq.io.lqCancelCnt <> io.lqCancelCnt
  lsq.io.sqCancelCnt <> io.sqCancelCnt
  // 把lsq中的Deq信号(提交个数)送到ctrlBlock
  // ctrlBlock基于此决定dispatch个数等信息
  lsq.io.lqDeq <> io.lqDeq
  lsq.io.sqDeq <> io.sqDeq
=======
>>>>>>> Stashed changes
  lsq.io.lqCancelCnt <> io.mem_to_ooo.lqCancelCnt
  lsq.io.sqCancelCnt <> io.mem_to_ooo.sqCancelCnt
  lsq.io.lqDeq <> io.mem_to_ooo.lqDeq
  lsq.io.sqDeq <> io.mem_to_ooo.sqDeq
  lsq.io.tl_d_channel <> dcache.io.lsu.tl_d_channel

<<<<<<< Updated upstream
  // LSQ to store buffer
  lsq.io.sbuffer        <> sbuffer.io.in
=======
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
  // LSQ to store buffer
  // 从store queue写出的数据给sbuffer
  lsq.io.sbuffer        <> sbuffer.io.in
  // 从store queue给sbuffer信号, 表示store queue是否为空
>>>>>>> Stashed changes
  lsq.io.sqEmpty        <> sbuffer.io.sqempty
  dcache.io.force_write := lsq.io.force_write
  // Sbuffer
  sbuffer.io.csrCtrl    <> csrCtrl
  sbuffer.io.dcache     <> dcache.io.lsu.store
  sbuffer.io.force_write <> lsq.io.force_write
  // flush sbuffer
<<<<<<< Updated upstream
  val fenceFlush = io.ooo_to_mem.flushSb
=======
<<<<<<< HEAD
  // fence执行时需要flush store buffer
  val fenceFlush = io.fenceToSbuffer.flushSb
  // atomicUnits执行时, 需要flush sbuffer
=======
  val fenceFlush = io.ooo_to_mem.flushSb
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes
  val atomicsFlush = atomicsUnit.io.flush_sbuffer.valid
  val stIsEmpty = sbuffer.io.flush.empty && uncache.io.flush.empty
  io.mem_to_ooo.sbIsEmpty := RegNext(stIsEmpty)

  // if both of them tries to flush sbuffer at the same time
  // something must have gone wrong
  assert(!(fenceFlush && atomicsFlush))
  sbuffer.io.flush.valid := RegNext(fenceFlush || atomicsFlush)
  uncache.io.flush.valid := sbuffer.io.flush.valid

  // Vector Load/Store Queue
  vlsq.io.int2vlsu <> io.int2vlsu
  vlsq.io.vec2vlsu <> io.vec2vlsu
  vlsq.io.vlsu2vec <> io.vlsu2vec
  vlsq.io.vlsu2int <> io.vlsu2int
  vlsq.io.vlsu2ctrl <> io.vlsu2ctrl

  // AtomicsUnit: AtomicsUnit will override other control signials,
  // as atomics insts (LR/SC/AMO) will block the pipeline
  val s_normal +: s_atomics = Enum(exuParameters.StuCnt + 1)
  val state = RegInit(s_normal)

  val atomic_rs = (0 until exuParameters.StuCnt).map(exuParameters.LduCnt + _)
  val atomic_replay_port_idx = (0 until exuParameters.StuCnt)
  val st_atomics = Seq.tabulate(exuParameters.StuCnt)(i =>
    io.ooo_to_mem.issue(atomic_rs(i)).valid && FuType.storeIsAMO((io.ooo_to_mem.issue(atomic_rs(i)).bits.uop.ctrl.fuType))
  )

  val st_data_atomics = Seq.tabulate(exuParameters.StuCnt)(i =>
    stData(i).valid && FuType.storeIsAMO(stData(i).bits.uop.ctrl.fuType)
  )

  // TODO: 这里是否可以按pipeline区分？
  for (i <- 0 until exuParameters.StuCnt) when(st_atomics(i)) {
    io.ooo_to_mem.issue(atomic_rs(i)).ready := atomicsUnit.io.in.ready
    storeUnits(i).io.stin.valid := false.B

    state := s_atomics(i)
    if (exuParameters.StuCnt > 1)
      assert(!st_atomics.zipWithIndex.filterNot(_._2 == i).unzip._1.reduce(_ || _))
  }
  when (atomicsUnit.io.out.valid) {
    assert((0 until exuParameters.StuCnt).map(state === s_atomics(_)).reduce(_ || _))
    state := s_normal
  }

  atomicsUnit.io.in.valid := st_atomics.reduce(_ || _)
  atomicsUnit.io.in.bits  := Mux1H(Seq.tabulate(exuParameters.StuCnt)(i =>
    st_atomics(i) -> io.ooo_to_mem.issue(atomic_rs(i)).bits))
  atomicsUnit.io.storeDataIn.valid := st_data_atomics.reduce(_ || _)
  atomicsUnit.io.storeDataIn.bits  := Mux1H(Seq.tabulate(exuParameters.StuCnt)(i =>
    st_data_atomics(i) -> stData(i).bits))
  atomicsUnit.io.rsIdx    := Mux1H(Seq.tabulate(exuParameters.StuCnt)(i =>
    st_atomics(i) -> io.rsfeedback(atomic_rs(i)).rsIdx))
  atomicsUnit.io.redirect <> redirect

  // TODO: complete amo's pmp support
  val amoTlb = dtlb_ld(0).requestor(0)
  atomicsUnit.io.dtlb.resp.valid := false.B
  atomicsUnit.io.dtlb.resp.bits  := DontCare
  atomicsUnit.io.dtlb.req.ready  := amoTlb.req.ready
  atomicsUnit.io.pmpResp := pmp_check(0).resp

  atomicsUnit.io.dcache <> dcache.io.lsu.atomics
  atomicsUnit.io.flush_sbuffer.empty := stIsEmpty

  atomicsUnit.io.csrCtrl := csrCtrl

  // for atomicsUnit, it uses loadUnit(0)'s TLB port

  when (state =/= s_normal) {
    // use store wb port instead of load
    loadUnits(0).io.ldout.ready := false.B
    // use load_0's TLB
    atomicsUnit.io.dtlb <> amoTlb

    // hw prefetch should be disabled while executing atomic insts
    loadUnits.map(i => i.io.prefetch_req.valid := false.B)

    // make sure there's no in-flight uops in load unit
    assert(!loadUnits(0).io.ldout.valid)
  }

  for (i <- 0 until exuParameters.StuCnt) when (state === s_atomics(i)) {
    atomicsUnit.io.feedbackSlow <> io.rsfeedback(atomic_rs(i)).feedbackSlow

    assert(!storeUnits(i).io.feedback_slow.valid)
  }

  lsq.io.exceptionAddr.isStore := io.ooo_to_mem.isStore
  // Exception address is used several cycles after flush.
  // We delay it by 10 cycles to ensure its flush safety.
  val atomicsException = RegInit(false.B)
  when (DelayN(redirect.valid, 10) && atomicsException) {
    atomicsException := false.B
  }.elsewhen (atomicsUnit.io.exceptionAddr.valid) {
    atomicsException := true.B
  }
  val atomicsExceptionAddress = RegEnable(atomicsUnit.io.exceptionAddr.bits, atomicsUnit.io.exceptionAddr.valid)
<<<<<<< Updated upstream
  io.mem_to_ooo.lsqio.vaddr := RegNext(Mux(atomicsException, atomicsExceptionAddress, lsq.io.exceptionAddr.vaddr))
=======
<<<<<<< HEAD
  // 把异常地址送出给csr, 准备异常处理
  io.lsqio.exceptionAddr.vaddr := RegNext(Mux(atomicsException, atomicsExceptionAddress, lsq.io.exceptionAddr.vaddr))
  // TODO: 为什么atomic有异常需要处理时,不允许有新指令进入?
=======
  io.mem_to_ooo.lsqio.vaddr := RegNext(Mux(atomicsException, atomicsExceptionAddress, lsq.io.exceptionAddr.vaddr))
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
>>>>>>> Stashed changes
  XSError(atomicsException && atomicsUnit.io.in.valid, "new instruction before exception triggers\n")

  io.memInfo.sqFull := RegNext(lsq.io.sqFull)
  io.memInfo.lqFull := RegNext(lsq.io.lqFull)
  io.memInfo.dcacheMSHRFull := RegNext(dcache.io.mshrFull)

  val ldDeqCount = PopCount(io.ooo_to_mem.issue.take(exuParameters.LduCnt).map(_.valid))
  val stDeqCount = PopCount(io.ooo_to_mem.issue.drop(exuParameters.LduCnt).map(_.valid))
  val rsDeqCount = ldDeqCount + stDeqCount
  XSPerfAccumulate("load_rs_deq_count", ldDeqCount)
  XSPerfHistogram("load_rs_deq_count", ldDeqCount, true.B, 0, exuParameters.LduCnt, 1)
  XSPerfAccumulate("store_rs_deq_count", stDeqCount)
  XSPerfHistogram("store_rs_deq_count", stDeqCount, true.B, 0, exuParameters.StuCnt, 1)
  XSPerfAccumulate("ls_rs_deq_count", rsDeqCount)

  val pfevent = Module(new PFEvent)
  pfevent.io.distribute_csr := csrCtrl.distribute_csr
  val csrevents = pfevent.io.hpmevent.slice(16,24)

  val memBlockPerfEvents = Seq(
    ("ldDeqCount", ldDeqCount),
    ("stDeqCount", stDeqCount),
  )
  val allPerfEvents = memBlockPerfEvents ++ (loadUnits ++ Seq(sbuffer, lsq, dcache)).flatMap(_.getPerfEvents)
  val hpmEvents = allPerfEvents.map(_._2.asTypeOf(new PerfEvent)) ++ perfEventsPTW
  val perfEvents = HPerfMonitor(csrevents, hpmEvents).getPerfEvents
  generatePerfEvent()
}