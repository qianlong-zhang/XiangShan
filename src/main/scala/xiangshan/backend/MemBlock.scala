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
import xiangshan.mem.prefetch.{BasePrefecher, SMSParams, SMSPrefetcher, L1Prefetcher}

class Std(implicit p: Parameters) extends FunctionUnit {
  io.in.ready := true.B
  io.out.valid := io.in.valid
  io.out.bits.uop := io.in.bits.uop
  io.out.bits.data := io.in.bits.src(0)
}

class ooo_to_mem(implicit p: Parameters) extends XSBundle {
  val loadFastMatch = Vec(exuParameters.LduCnt, Input(UInt(exuParameters.LduCnt.W)))
  val loadFastFuOpType = Vec(exuParameters.LduCnt, Input(FuOpType()))
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
  val storePc = Vec(exuParameters.StuCnt, Input(UInt(VAddrBits.W))) // for hw prefetch
  val issue = Vec(exuParameters.LsExuCnt + exuParameters.StuCnt, Flipped(DecoupledIO(new ExuInput)))
}

class mem_to_ooo(implicit p: Parameters ) extends XSBundle {
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

class MemCoreTopDownIO extends Bundle {
  val robHeadMissInDCache = Output(Bool())
  val robHeadTlbReplay = Output(Bool())
  val robHeadTlbMiss = Output(Bool())
  val robHeadLoadVio = Output(Bool())
  val robHeadLoadMSHR = Output(Bool())
}

class fetch_to_mem(implicit p: Parameters) extends XSBundle{
  val itlb = Flipped(new TlbPtwIO())
}


class MemBlock()(implicit p: Parameters) extends LazyModule
  with HasXSParameter with HasWritebackSource {
  override def shouldBeInlined: Boolean = false

  val dcache = LazyModule(new DCacheWrapper())
  val uncache = LazyModule(new Uncache())
  val ptw = LazyModule(new L2TLBWrapper())
  val ptw_to_l2_buffer = if (!coreParams.softPTW) LazyModule(new TLBuffer) else null
  val l2_pf_sender_opt = coreParams.prefetcher.map(_ =>
    BundleBridgeSource(() => new PrefetchRecv)
  )
  val l3_pf_sender_opt = coreParams.prefetcher.map(_ =>
    BundleBridgeSource(() => new huancun.PrefetchRecv)
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
  with HasL1PrefetchSourceParameter
{

  val io = IO(new Bundle {
    val hartId = Input(UInt(8.W))
    val redirect = Flipped(ValidIO(new Redirect))

    val ooo_to_mem = new ooo_to_mem
    val mem_to_ooo = new mem_to_ooo
    val fetch_to_mem = new fetch_to_mem

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
    val l2PfqBusy = Input(Bool())

    val debugTopDown = new Bundle {
      val robHeadVaddr = Flipped(Valid(UInt(VAddrBits.W)))
      val toCore = new MemCoreTopDownIO
    }
  })

  override def writebackSource1: Option[Seq[Seq[DecoupledIO[ExuOutput]]]] = Some(Seq(io.mem_to_ooo.writeback))

  val redirect = RegNextWithEnable(io.redirect)

  val dcache = outer.dcache.module
  val uncache = outer.uncache.module

  // dcache给lsq返回数据, 打一拍
  val delayedDcacheRefill = RegNext(dcache.io.lsu.lsq)

  //TODO: 为什么delay2? 分布式的csrCtrl是不是需要2个cycle才能拿到?
  val csrCtrl = DelayN(io.ooo_to_mem.csrCtrl, 2)
  dcache.io.csr.distribute_csr <> csrCtrl.distribute_csr
  // TODO: 这里为什么延迟数有区别?
  dcache.io.l2_pf_store_only := RegNext(io.ooo_to_mem.csrCtrl.l2_pf_store_only, false.B)
  io.mem_to_ooo.csrUpdate := RegNext(dcache.io.csr.update)
  io.error <> RegNext(RegNext(dcache.io.error))
  when(!csrCtrl.cache_error_enable){
    io.error.report_to_beu := false.B
    io.error.valid := false.B
  }

  val loadUnits = Seq.fill(exuParameters.LduCnt)(Module(new LoadUnit))
  val storeUnits = Seq.fill(exuParameters.StuCnt)(Module(new StoreUnit))
  val stdExeUnits = Seq.fill(exuParameters.StuCnt)(Module(new StdExeUnit))
  // 获取store指令的data输出, 发送给lsq与atomic
  val stData = stdExeUnits.map(_.io.out)
  // 不包含std
  val exeUnits = loadUnits ++ storeUnits
  val l1_pf_req = Wire(Decoupled(new L1PrefetchReq()))
  // TODO: 送给预取器的信号为什么都是延迟2拍?
  val prefetcherOpt: Option[BasePrefecher] = coreParams.prefetcher.map {
    case _: SMSParams =>
      val sms = Module(new SMSPrefetcher())
      // 把配置SMS的信息通过csr写入SMS中
      sms.io_agt_en := RegNextN(io.ooo_to_mem.csrCtrl.l1D_pf_enable_agt, 2, Some(false.B))
      sms.io_pht_en := RegNextN(io.ooo_to_mem.csrCtrl.l1D_pf_enable_pht, 2, Some(false.B))
      sms.io_act_threshold := RegNextN(io.ooo_to_mem.csrCtrl.l1D_pf_active_threshold, 2, Some(12.U))
      sms.io_act_stride := RegNextN(io.ooo_to_mem.csrCtrl.l1D_pf_active_stride, 2, Some(30.U))
      sms.io_stride_en := false.B
      sms
  }
  prefetcherOpt.foreach{ pf => pf.io.l1_req.ready := false.B }
  val l1PrefetcherOpt: Option[BasePrefecher] = coreParams.prefetcher.map {
    case _ =>
      val l1Prefetcher = Module(new L1Prefetcher())
      l1Prefetcher.io.enable := WireInit(Constantin.createRecord("enableL1StreamPrefetcher" + p(XSCoreParamsKey).HartId.toString, initValue = 1.U)) === 1.U
      l1Prefetcher.pf_ctrl <> dcache.io.pf_ctrl
      l1Prefetcher.l2PfqBusy := io.l2PfqBusy

      // stride will train on miss or prefetch hit
      for (i <- 0 until exuParameters.LduCnt) {
        val source = loadUnits(i).io.prefetch_train_l1
        l1Prefetcher.stride_train(i).valid := source.valid && source.bits.isFirstIssue && (
          source.bits.miss || isFromStride(source.bits.meta_prefetch)
        )
        l1Prefetcher.stride_train(i).bits := source.bits
        l1Prefetcher.stride_train(i).bits.uop.cf.pc := Mux(loadUnits(i).io.s2_ptr_chasing, io.ooo_to_mem.loadPc(i), RegNext(io.ooo_to_mem.loadPc(i)))
      }
      l1Prefetcher
  }
  // load prefetch to l1 Dcache
  l1PrefetcherOpt match {
    case Some(pf) => l1_pf_req <> pf.io.l1_req
    case None =>
      l1_pf_req.valid := false.B
      l1_pf_req.bits := DontCare
  }
  //默认使能预取器在发生hit时继续训练
  val pf_train_on_hit = RegNextN(io.ooo_to_mem.csrCtrl.l1D_pf_train_on_hit, 2, Some(true.B))

  loadUnits.zipWithIndex.map(x => x._1.suggestName("LoadUnit_"+x._2))
  storeUnits.zipWithIndex.map(x => x._1.suggestName("StoreUnit_"+x._2))
  val atomicsUnit = Module(new AtomicsUnit)

  // Atom inst comes from sta / std, then its result
  // will be writebacked using load writeback port
  //
  // However, atom exception will be writebacked to rob
  // using store writeback port

  // TODO: atomic写回端口占用了loadUnits.head的写回端口, 那么lodUnits.head的写回怎么办? 直接丢弃会不会出错?
  // atomic指令会清空rob才dispatch，因此不会出现上述情况
  // 为什么atomic的异常是从store写回？ 因为atomic指令(例如lr)失败与否与store指令(例如sr)的执行结果相关， 与load的执行结果无关。
  val loadWritebackOverride  = Mux(atomicsUnit.io.out.valid, atomicsUnit.io.out.bits, loadUnits.head.io.ldout.bits)
  val ldout0 = Wire(Decoupled(new ExuOutput))
  ldout0.valid := atomicsUnit.io.out.valid || loadUnits.head.io.ldout.valid
  ldout0.bits  := loadWritebackOverride
  atomicsUnit.io.out.ready := ldout0.ready
  loadUnits.head.io.ldout.ready := ldout0.ready
>>>>>>> ffc9de54938a9574f465b83a71d5252cfd37cf30
  when(atomicsUnit.io.out.valid){
    // 如果写回的是atomicsUnit, 则atmoicsUnit的异常信号是从store写回, 而不是从loadUnits写回
    // 所以这里把loadUnits的异常向量全部清空
    // TODO: 为什么atomic的异常是从store写回？ 因为atomic指令(例如lr)失败与否与store指令(例如sc)的执行结果相关， 与load的执行结果无关。
    ldout0.bits.uop.cf.exceptionVec := 0.U(16.W).asBools // exception will be writebacked via store wb port
  }

  val ldExeWbReqs = ldout0 +: loadUnits.tail.map(_.io.ldout)
  io.mem_to_ooo.writeback <> ldExeWbReqs ++ VecInit(storeUnits.map(_.io.stout)) ++ VecInit(stdExeUnits.map(_.io.out))
  io.mem_to_ooo.otherFastWakeup := DontCare
  io.mem_to_ooo.otherFastWakeup.take(2).zip(loadUnits.map(_.io.fast_uop)).foreach{case(a,b)=> a := b}
  // TODO: 这里是硬编码, 获取loadUnits, 应该改为take(exuParameters.LduCnt)
  // 把从loadUnits中送出的fastUop信号通过MemBlock送到顶层的XSCore中
  // XSCore会把这些信号再送回exuBlocks中唤醒相应指令继续执行
  // io.otherFastWakeup.take(2).zip(loadUnits.map(_.io.fastUop)).foreach { case (a, b) => a := b }
  // 丢掉load和std的写回, 只剩sta的写回, 用于后续trigger和mmio的判断
  val stOut = io.mem_to_ooo.writeback.drop(exuParameters.LduCnt).dropRight(exuParameters.StuCnt)

  // prefetch to l1 req
  // Stream's confidence is always 1
  loadUnits.foreach(load_unit => {
    // 把预取器中输出的预取请求发给loadUnits, 在stage0会按照优先级挑选发送到后续load流水
    // 预取器中输出的请求是l1_pf_req
    load_unit.io.prefetch_req.valid <> l1_pf_req.valid
    load_unit.io.prefetch_req.bits <> l1_pf_req.bits
  })
  // NOTE: loadUnits(0) has higher bank conflict and miss queue arb priority than loadUnits(1)
  // when loadUnits(0) stage 0 is busy, hw prefetch will never use that pipeline
  val LowConfPort = 0
  // TODO: 送给lsu(0)的prefetch_req初始化confidence是0, 保证lsu中的正常请求不会被stall
  loadUnits(LowConfPort).io.prefetch_req.bits.confidence := 0.U

  // 只有在预取器发出的预取请求confidence高, 或者所有loadUnits中的loadIn都无效时,
  // 才发出l1_pf_req, 为了保证prefetch请求不影响正常load指令执行
  l1_pf_req.ready := (0 until exuParameters.LduCnt).map{
    case i => {
      if(i == LowConfPort) {
        loadUnits(i).io.canAcceptLowConfPrefetch
      }else {
        Mux(l1_pf_req.bits.confidence === 1.U, loadUnits(i).io.canAcceptHighConfPrefetch, loadUnits(i).io.canAcceptLowConfPrefetch)
      }
    }
  }.reduce(_ || _)

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
  // store queue中issue指针, 输出给MemBlock, 最终送到执行单元
  // 如果配置了checkWait则只有当其他指令在这个store后面时, 才允许发射
  io.mem_to_ooo.stIssuePtr := lsq.io.issuePtrExt

  // 把线程id送给各个模块
  dcache.io.hartId := io.hartId
  lsq.io.hartId := io.hartId
  sbuffer.io.hartId := io.hartId
  atomicsUnit.io.hartId := io.hartId

  dcache.io.lqEmpty := lsq.io.lqEmpty

  // load/store prefetch to l2 cache
  prefetcherOpt.foreach(sms_pf => {
    l1PrefetcherOpt.foreach(l1_pf => {
      val sms_pf_to_l2 = ValidIODelay(sms_pf.io.l2_req, 2)
      val l1_pf_to_l2 = ValidIODelay(l1_pf.io.l2_req, 2)

      outer.l2_pf_sender_opt.get.out.head._1.addr_valid := sms_pf_to_l2.valid || l1_pf_to_l2.valid
      outer.l2_pf_sender_opt.get.out.head._1.addr := Mux(l1_pf_to_l2.valid, l1_pf_to_l2.bits.addr, sms_pf_to_l2.bits.addr)
      outer.l2_pf_sender_opt.get.out.head._1.pf_source := Mux(l1_pf_to_l2.valid, l1_pf_to_l2.bits.source, sms_pf_to_l2.bits.source)
      outer.l2_pf_sender_opt.get.out.head._1.l2_pf_en := RegNextN(io.ooo_to_mem.csrCtrl.l2_pf_enable, 2, Some(true.B))

      sms_pf.io.enable := RegNextN(io.ooo_to_mem.csrCtrl.l1D_pf_enable, 2, Some(false.B))

      val l2_trace = Wire(new LoadPfDbBundle)
      l2_trace.paddr := outer.l2_pf_sender_opt.get.out.head._1.addr
      val table = ChiselDB.createTable("L2PrefetchTrace"+ p(XSCoreParamsKey).HartId.toString, new LoadPfDbBundle, basicDB = false)
      table.log(l2_trace, l1_pf_to_l2.valid, "StreamPrefetchTrace", clock, reset)
      table.log(l2_trace, !l1_pf_to_l2.valid && sms_pf_to_l2.valid, "L2PrefetchTrace", clock, reset)

      val l1_pf_to_l3 = ValidIODelay(l1_pf.io.l3_req, 4)
      outer.l3_pf_sender_opt.get.out.head._1.addr_valid := l1_pf_to_l3.valid
      outer.l3_pf_sender_opt.get.out.head._1.addr := l1_pf_to_l3.bits
      outer.l3_pf_sender_opt.get.out.head._1.l2_pf_en := RegNextN(io.ooo_to_mem.csrCtrl.l2_pf_enable, 4, Some(true.B))

      val l3_trace = Wire(new LoadPfDbBundle)
      l3_trace.paddr := outer.l3_pf_sender_opt.get.out.head._1.addr
      val l3_table = ChiselDB.createTable("L3PrefetchTrace"+ p(XSCoreParamsKey).HartId.toString, new LoadPfDbBundle, basicDB = false)
      l3_table.log(l3_trace, l1_pf_to_l3.valid, "StreamPrefetchTrace", clock, reset)

      XSPerfAccumulate("prefetch_fire_l2", outer.l2_pf_sender_opt.get.out.head._1.addr_valid)
      XSPerfAccumulate("prefetch_fire_l3", outer.l3_pf_sender_opt.get.out.head._1.addr_valid)
      XSPerfAccumulate("l1pf_fire_l2", l1_pf_to_l2.valid)
      XSPerfAccumulate("sms_fire_l2", !l1_pf_to_l2.valid && sms_pf_to_l2.valid)
      XSPerfAccumulate("sms_block_by_l1pf", l1_pf_to_l2.valid && sms_pf_to_l2.valid)
    })
  })

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
  val dtlb_ld = VecInit(Seq.fill(1){
    val tlb_ld = Module(new TLBNonBlock(exuParameters.LduCnt + 1, 2, ldtlbParams))
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
  // 展平所有dtlb的requestor, 生成一个新的Seq
  // requestor中信号包含req/req_kill/response
  val ptwio = Wire(new VectorTlbPtwIO(exuParameters.LduCnt + exuParameters.StuCnt + 2)) // load + store + hw prefetch
  val dtlb_reqs = dtlb.map(_.requestor).flatten
  val dtlb_pmps = dtlb.map(_.pmp).flatten
  // 把外部传入的sfence tlbcsr flushPipe信号传入dtlb中进行flush等处理
  dtlb.map(_.sfence := sfence)
  dtlb.map(_.csr := tlbcsr)
  dtlb.map(_.flushPipe.map(a => a := false.B)) // non-block doesn't need
  // TODO: 当前refillBothTlb是false, outReplace也是false, 如果下面代码中if都不成立, 那么当前tlb替换时, 如何选择的victim?
  // 把返回的ptw vpn填入TLB中
  if (refillBothTlb) {
    require(ldtlbParams.outReplace == sttlbParams.outReplace)
    require(ldtlbParams.outReplace == pftlbParams.outReplace)
    require(ldtlbParams.outReplace)

    val replace = Module(new TlbReplace(exuParameters.LduCnt + exuParameters.StuCnt + 2, ldtlbParams))
    replace.io.apply_sep(dtlb_ld.map(_.replace) ++ dtlb_st.map(_.replace) ++ dtlb_prefetch.map(_.replace), ptwio.resp.bits.data.entry.tag)
  } else {
    if (ldtlbParams.outReplace) {
      val replace_ld = Module(new TlbReplace(exuParameters.LduCnt, ldtlbParams))
      replace_ld.io.apply_sep(dtlb_ld.map(_.replace), ptwio.resp.bits.data.entry.tag)
    }
    if (sttlbParams.outReplace) {
      val replace_st = Module(new TlbReplace(exuParameters.StuCnt, sttlbParams))
      replace_st.io.apply_sep(dtlb_st.map(_.replace), ptwio.resp.bits.data.entry.tag)
    }
    if (pftlbParams.outReplace) {
      val replace_pf = Module(new TlbReplace(1, pftlbParams))
      replace_pf.io.apply_sep(dtlb_prefetch.map(_.replace), ptwio.resp.bits.data.entry.tag)
    }
  }

  val ptw_resp_next = RegEnable(ptwio.resp.bits, ptwio.resp.valid)
  // 只有在sfence没执行或satp没有改变情况下ptw的response才有效
  val ptw_resp_v = RegNext(ptwio.resp.valid && !(sfence.valid && tlbcsr.satp.changed), init = false.B)
  ptwio.resp.ready := true.B

  // 把dtlb中TLB对象中的ptw.req与MemBlockImp中的io.ptw.req连起来。
  dtlb.flatMap(a => a.ptw.req)
    .zipWithIndex
    // 这里的tlb代表每个dtlb中的ptw.req成员变量
    .foreach{ case (tlb, i) =>
      // 是否需要真的发出ptw请求? 只有在dtlb中ptw.req有效, 且满足如下条件时才发出:
      // ptw_resp_v为假或者vector_hit为假或者ptw_resp_next没有hit
      // 也就是当拍没有ptw的response, 且当拍不会回填ptw时才会发出ptw请求
      tlb.ready := ptwio.req(i).ready
      ptwio.req(i).bits := tlb.bits
    val vector_hit = if (refillBothTlb) Cat(ptw_resp_next.vector).orR
      else if (i < (exuParameters.LduCnt + 1)) Cat(ptw_resp_next.vector.take(exuParameters.LduCnt + 1)).orR
      else if (i < (exuParameters.LduCnt + 1 + exuParameters.StuCnt)) Cat(ptw_resp_next.vector.drop(exuParameters.LduCnt + 1)).orR
      else Cat(ptw_resp_next.vector.drop(exuParameters.LduCnt + exuParameters.StuCnt + 1)).orR
    ptwio.req(i).valid := tlb.valid && !(ptw_resp_v && vector_hit &&
      ptw_resp_next.data.hit(tlb.bits.vpn, tlbcsr.satp.asid, allType = true, ignoreAsid = true))
  }
  // 把从MemBlock接收到的response, 打一拍后(也就是ptw_resp_next)给dtlb
  dtlb.foreach(_.ptw.resp.bits := ptw_resp_next.data)
  if (refillBothTlb) {
    // ptw_resp有效情况下, 且两个loadUnit, 两个StoreUnit, 一个Prefetch,任意一个有有效的response数据就回填
    dtlb.foreach(_.ptw.resp.valid := ptw_resp_v && Cat(ptw_resp_next.vector).orR)
  } else {
    // load的dtlb是否收到了有效的ptw.resp? 只有在resp有效, 且resp的vector中对应load的有效时才说明ptw的resp有效
    // 把该信号赋值给各个dtlb
    dtlb_ld.foreach(_.ptw.resp.valid := ptw_resp_v && Cat(ptw_resp_next.vector.take(exuParameters.LduCnt + 1)).orR)
    dtlb_st.foreach(_.ptw.resp.valid := ptw_resp_v && Cat(ptw_resp_next.vector.drop(exuParameters.LduCnt + 1).take(exuParameters.StuCnt)).orR)
    dtlb_prefetch.foreach(_.ptw.resp.valid := ptw_resp_v && Cat(ptw_resp_next.vector.drop(exuParameters.LduCnt + exuParameters.StuCnt + 1)).orR)
  }

  val dtlbRepeater1  = PTWFilter(ldtlbParams.fenceDelay, ptwio, sfence, tlbcsr, l2tlbParams.dfilterSize)
  val dtlbRepeater2  = PTWRepeaterNB(passReady = false, ldtlbParams.fenceDelay, dtlbRepeater1.io.ptw, ptw.io.tlb(1), sfence, tlbcsr)
  val itlbRepeater2 = PTWRepeaterNB(passReady = false, itlbParams.fenceDelay, io.fetch_to_mem.itlb, ptw.io.tlb(0), sfence, tlbcsr)

  lsq.io.debugTopDown.robHeadMissInDTlb := dtlbRepeater1.io.rob_head_miss_in_tlb

  // pmp
  val pmp = Module(new PMP())
  // csr是分布式的, 当指令更新csr时, 需要同步这些所有分布式的csr
  pmp.io.distribute_csr <> csrCtrl.distribute_csr

  // 针对每个load/store/prefetcher unit都单独有一个PMPChecker
  // 把dtlb的pmp请求连接到PMPChecker
  val pmp_check = VecInit(Seq.fill(exuParameters.LduCnt + exuParameters.StuCnt + 2)(Module(new PMPChecker(3)).io))
  for ((p,d) <- pmp_check zip dtlb_pmps) {
    p.apply(tlbcsr.priv.dmode, pmp.io.pmp, pmp.io.pma, d)
    require(p.req.bits.size.getWidth == d.bits.size.getWidth)
  }

  for (i <- 0 until exuParameters.LduCnt) {
    io.debug_ls.debugLsInfo(i) := loadUnits(i).io.debug_ls
  }
  for (i <- 0 until exuParameters.StuCnt) {
    io.debug_ls.debugLsInfo(i + exuParameters.LduCnt) := storeUnits(i).io.debug_ls
  }

  io.mem_to_ooo.lsTopdownInfo := loadUnits.map(_.io.lsTopdownInfo)

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

  // 对于发生Fast Replay的请求, 需要根据年龄确定replay顺序
  // 针对每一个ldu, 生成一个BanlanceEntry, 把这些entry封装成Seq传给balanceReOrder
  // 然后输出一个重排序后的fastReplay顺序, 对于ldu(0), 只选择port = 0的
  val fastReplaySel = loadUnits.zipWithIndex.map { case (ldu, i) => {
    val wrapper = Wire(Valid(new BalanceEntry))
    wrapper.valid        := ldu.io.fast_rep_out.valid
    wrapper.bits.req     := ldu.io.fast_rep_out.bits
    wrapper.bits.balance := ldu.io.fast_rep_out.bits.rep_info.bank_conflict
    wrapper.bits.port    := i.U
    wrapper
  }}
  val balanceFastReplaySel = balanceReOrder(fastReplaySel)

  val correctMissTrain = WireInit(Constantin.createRecord("CorrectMissTrain" + p(XSCoreParamsKey).HartId.toString, initValue = 0.U)) === 1.U

  for (i <- 0 until exuParameters.LduCnt) {
    loadUnits(i).io.redirect <> redirect
    loadUnits(i).io.isFirstIssue := true.B

    // get input form dispatch
    loadUnits(i).io.ldin <> io.ooo_to_mem.issue(i)
    loadUnits(i).io.feedback_slow <> io.rsfeedback(i).feedbackSlow
    loadUnits(i).io.feedback_fast <> io.rsfeedback(i).feedbackFast
    loadUnits(i).io.rsIdx := io.rsfeedback(i).rsIdx
    loadUnits(i).io.correctMissTrain := correctMissTrain

    // fast replay
    loadUnits(i).io.fast_rep_in.valid := balanceFastReplaySel(i).valid
    loadUnits(i).io.fast_rep_in.bits := balanceFastReplaySel(i).bits.req

    loadUnits(i).io.fast_rep_out.ready := false.B
    for (j <- 0 until exuParameters.LduCnt) {
      // 如果通过balanceReOrder的输出有效, 且端口能匹配, 则把传递给loadUnits的fastReplay信号拉高, 表示真的有fastReplay
      // 注意这里的i/j不同. j=1, i=0: 如果balanceFastReplaySel(1).valid且balanceFastReplaySel(1).bits.port = 0
      // 则把loadUnits(0).io.fastReplayOut.ready = loadUnits(1).io.fastReplayIn.ready
      when (balanceFastReplaySel(j).valid && balanceFastReplaySel(j).bits.port === i.U) {
        loadUnits(i).io.fast_rep_out.ready := loadUnits(j).io.fast_rep_in.ready
      }
    }

    // get input form dispatch
    loadUnits(i).io.ldin <> io.ooo_to_mem.issue(i)
    // dcache access
    // dcache的访问请求是loadUnits中S0发生的, 把该信号送给dcache去处理
    loadUnits(i).io.dcache <> dcache.io.lsu.load(i)
    // forward
    // 从lsq/sbuffer中获取forward数据
    loadUnits(i).io.lsq.forward <> lsq.io.forward(i)
    loadUnits(i).io.sbuffer <> sbuffer.io.forward(i)
    // 在发出Tilink的命令是TLMessages.GrantData时, dcache可以forward给loadUnits数据
    loadUnits(i).io.tl_d_channel := dcache.io.lsu.forward_D(i)
    loadUnits(i).io.forward_mshr <> dcache.io.lsu.forward_mshr(i)
    // ld-ld violation check
    // load的S2会去check是否发生了violation, 把信号送到lsq去确认
    loadUnits(i).io.lsq.ldld_nuke_query <> lsq.io.ldu.ldld_nuke_query(i)
    loadUnits(i).io.lsq.stld_nuke_query <> lsq.io.ldu.stld_nuke_query(i)
    loadUnits(i).io.csrCtrl       <> csrCtrl
    // dcache refill req
    // 从dcache来的数据, 同时送给loadUnits
    // TODO: Perf: 这里loadUnits流水线中没有去用, 可以改一下
    loadUnits(i).io.refill           <> delayedDcacheRefill
    // dtlb
    // 把loadUnits中访问tlb的请求送到tlb去处理
    loadUnits(i).io.tlb <> dtlb_reqs.take(exuParameters.LduCnt)(i)
    // pmp
    // 把loadUnits中访问pmp的请求送到pmp去检查，返回resp
    loadUnits(i).io.pmp <> pmp_check(i).resp
    // st-ld violation query
    for (s <- 0 until StorePipelineWidth) {
      //store S1输出给ldu, 看看有些load是否需要重新执行
      // store流水线中执行的指令, 送到loadUnits中进行检查, 是否发生violation
      loadUnits(i).io.stld_nuke_query(s) := storeUnits(s).io.stld_nuke_query
    }
    // 如果lsq满了, 反馈信号给loadUnits, 用于触发fastReplay
    loadUnits(i).io.lq_rep_full <> lsq.io.lq_rep_full
    // load prefetch train
    prefetcherOpt.foreach(pf => {
      // sms will train on all miss load sources
      val source = loadUnits(i).io.prefetch_train
      pf.io.ld_in(i).valid := Mux(pf_train_on_hit,
        source.valid,
        source.valid && source.bits.isFirstIssue && source.bits.miss
      )
      pf.io.ld_in(i).bits := source.bits
      pf.io.ld_in(i).bits.uop.cf.pc := Mux(loadUnits(i).io.s2_ptr_chasing, io.ooo_to_mem.loadPc(i), RegNext(io.ooo_to_mem.loadPc(i)))
    })
    l1PrefetcherOpt.foreach(pf => {
      // stream will train on all load sources
      val source = loadUnits(i).io.prefetch_train_l1
      pf.io.ld_in(i).valid := source.valid && source.bits.isFirstIssue
      pf.io.ld_in(i).bits := source.bits
      pf.io.st_in(i).valid := false.B
      pf.io.st_in(i).bits := DontCare
    })

    // load to load fast forward: load(i) prefers data(i)
    // 两条load流水线, 可能构成数据互相依赖, 如果有这种情况, load pipe0优先从pipe0获取数据
    // TODO: 如果同时有load data返回, 为什么优先从当前load pipe获取数据? 是因为距离自己更近吗?
    // 这里的fastPriority值: 如果i=2，exuParameters.LduCnt=5，则fastPriority的值为Seq(2, 3, 4, 0, 1)
    // i=0, LduCnt=2, fastPriority = Seq(0,1) ++ null = Seq(0, 1)
    // i=1, LduCnt=2, fastPriority = Seq(1) ++ Seq(0) = Seq(1, 0)
    val fastPriority = (i until exuParameters.LduCnt) ++ (0 until i)
    // 结合上面, loadUnit0产生Seq(0, 1), fastValidVec就是(loadUnits(0).io.fastpathOut.valid, loadUnits(1).io.fastpathOut.valid)
    // loadUnit(1)的fastValidVec就是(loadUnits(1).io.fastpathOut.valid, loadUnits(0).io.fastpathOut.valid)
    // 假如: loadUnits(0).io.fastpathOut.valid = true.B;  loadUnits(1).io.fastpathOut.valid = false.B
    // fastValidVec = Seq(true.B, false.B);
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
    loadUnits(i).io.ld_fast_fuOpType := io.ooo_to_mem.loadFastFuOpType(i)
    loadUnits(i).io.replay <> lsq.io.replay(i)

    // TODO: 这个hint作用是什么? L2用来告诉loadUnits,是否要尽快发起replay
    // 最终信号来源是CustomL1Hint.scala中的l1Hint, 如果L1说3个cycle后就能获取数据, 那么loadUnits就尽快发起fastReplay
    loadUnits(i).io.l2_hint <> io.l2_hint

    // 把loadUnits和lsq连起来
    // passdown to lsq (load s2)
    // 从loadUnits写回数据到load Queue
    lsq.io.ldu.ldin(i) <> loadUnits(i).io.lsq.ldin
    lsq.io.ldout(i) <> loadUnits(i).io.lsq.uncache
    // loadUnits从loadQueue中拿Raw数据
    lsq.io.ld_raw_data(i) <> loadUnits(i).io.lsq.ld_raw_data
    // 把loadUnits和lsq的trigger连起来
    lsq.io.trigger(i) <> loadUnits(i).io.lsq.trigger

    // 把l2Hint送给lsq
    lsq.io.l2_hint.valid := io.l2_hint.valid
    lsq.io.l2_hint.bits.sourceId := io.l2_hint.bits.sourceId

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
<<<<<<< HEAD
  // PrefetcherDTLBPortIndex 正好是dtlb_reqs中对应dtlb_prefetch
  // 这里是把prefetcher的tlb_req送到pfetch专用的dtlb_prefetch去处理
  val PrefetcherDTLBPortIndex = exuParameters.LduCnt + exuParameters.StuCnt
=======
  val StreamDTLBPortIndex = exuParameters.LduCnt
  // PrefetcherDTLBPortIndex 正好是dtlb_reqs中对应dtlb_prefetch
  // 这里是把prefetcher的tlb_req送到pfetch专用的dtlb_prefetch去处理
  val PrefetcherDTLBPortIndex = exuParameters.LduCnt + exuParameters.StuCnt + 1
>>>>>>> c89b46421f4e4f58aeacd51297260c254a386e8b
  prefetcherOpt match {
  case Some(pf) => dtlb_reqs(PrefetcherDTLBPortIndex) <> pf.io.tlb_req
  case None =>
    dtlb_reqs(PrefetcherDTLBPortIndex) := DontCare
    dtlb_reqs(PrefetcherDTLBPortIndex).req.valid := false.B
    dtlb_reqs(PrefetcherDTLBPortIndex).resp.ready := true.B
  }
  l1PrefetcherOpt match {
    case Some(pf) => dtlb_reqs(StreamDTLBPortIndex) <> pf.io.tlb_req
    case None =>
        dtlb_reqs(StreamDTLBPortIndex) := DontCare
        dtlb_reqs(StreamDTLBPortIndex).req.valid := false.B
        dtlb_reqs(StreamDTLBPortIndex).resp.ready := true.B
  }

  // StoreUnit
  // StoreUnit, 分成sta(storeUnits(i))和std(stdExeUnits(i))分别处理
  for (i <- 0 until exuParameters.StuCnt) {
    val stu = storeUnits(i)

    // 连接store data 流水线
    stdExeUnits(i).io.redirect <> redirect
    // issue = LsExuCnt + StuCnt, 其中LsExuCnt = LduCnt + StuCnt
    // 展开后issue = LduCnt + StuCnt + StuCnt, 这里的i就是用来取出store data的
    stdExeUnits(i).io.fromInt <> io.ooo_to_mem.issue(i + exuParameters.LduCnt + exuParameters.StuCnt)
    stdExeUnits(i).io.fromFp := DontCare
    stdExeUnits(i).io.out := DontCare

    stu.io.redirect      <> redirect
<<<<<<< HEAD
    // 从stu输出到rsfeedback, 由于rsfeedback是和ldu一起计算, 因此前面加上LduCnt后表示是stu的开始
=======
    stu.io.dcache        <> dcache.io.lsu.sta(i)
>>>>>>> c89b46421f4e4f58aeacd51297260c254a386e8b
    stu.io.feedback_slow <> io.rsfeedback(exuParameters.LduCnt + i).feedbackSlow
    // 从rs输出的rsIdx
    stu.io.rsIdx         <> io.rsfeedback(exuParameters.LduCnt + i).rsIdx
    // NOTE: just for dtlb's perf cnt
    // 从rs输出的isFirstIssue
    stu.io.isFirstIssue <> io.rsfeedback(exuParameters.LduCnt + i).isFirstIssue
    // 从rs输入的uop
    stu.io.stin         <> io.ooo_to_mem.issue(exuParameters.LduCnt + i)
    // 从stu的s1 stage输出到lsq
    stu.io.lsq          <> lsq.io.sta.storeAddrIn(i)
    // 从stu的s2 stage输出到lsq
    stu.io.lsq_replenish <> lsq.io.sta.storeAddrInRe(i)
    // dtlb
<<<<<<< HEAD
    // 把sta的tlb请求送到专属于store的dtlb_st中处理
    stu.io.tlb          <> dtlb_reqs.drop(exuParameters.LduCnt)(i)
    // 从pmp返回的check结果给stu
    stu.io.pmp          <> pmp_check(i+exuParameters.LduCnt).resp
=======
    stu.io.tlb          <> dtlb_reqs.drop(exuParameters.LduCnt + 1)(i)
    stu.io.pmp          <> pmp_check(exuParameters.LduCnt + 1 + i).resp

    // prefetch
    stu.io.prefetch_req <> sbuffer.io.store_prefetch(i)
>>>>>>> c89b46421f4e4f58aeacd51297260c254a386e8b

    // store unit does not need fast feedback
    // storeunit没有快速唤醒通路
    io.rsfeedback(exuParameters.LduCnt + i).feedbackFast := DontCare

    // Lsq to sta unit
    // stu在S0输出给lsq的store mask
    lsq.io.sta.storeMaskIn(i) <> stu.io.st_mask_out

    // Lsq to std unit's rs
    // store data输出给lsq
    lsq.io.std.storeDataIn(i) := stData(i)

    // store prefetch train
    prefetcherOpt.foreach(pf => {
      pf.io.st_in(i).valid := Mux(pf_train_on_hit,
        stu.io.prefetch_train.valid,
        stu.io.prefetch_train.valid && stu.io.prefetch_train.bits.isFirstIssue && (
          stu.io.prefetch_train.bits.miss
          )
      )
      pf.io.st_in(i).bits := stu.io.prefetch_train.bits
      pf.io.st_in(i).bits.uop.cf.pc := RegNext(io.ooo_to_mem.storePc(i))
    })

    // 1. sync issue info to store set LFST
    // 2. when store issue, broadcast issued sqPtr to wake up the following insts
    // io.stIn(i).valid := io.issue(exuParameters.LduCnt + i).valid
    // io.stIn(i).bits := io.issue(exuParameters.LduCnt + i).bits
    // stu中指令发射后, 通过stIn发送到LFST去更新表中内容
    io.mem_to_ooo.stIn(i).valid := stu.io.issue.valid
    io.mem_to_ooo.stIn(i).bits := stu.io.issue.bits

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

    // when atom inst writeback, suppress normal load trigger
    (0 until exuParameters.LduCnt).map(i => {
      io.mem_to_ooo.writeback(i).bits.uop.cf.trigger.backendHit := VecInit(Seq.fill(6)(false.B))
    })
  }

  // Uncahce
  // 目前uncache的写不支持outstanding
  uncache.io.enableOutstanding := io.ooo_to_mem.csrCtrl.uncache_write_outstanding_enable
  io.mem_to_ooo.lsqio.mmio       := lsq.io.rob.mmio
  lsq.io.rob.lcommit             := io.ooo_to_mem.lsqio.lcommit
  lsq.io.rob.scommit             := io.ooo_to_mem.lsqio.scommit
  lsq.io.rob.pendingld           := io.ooo_to_mem.lsqio.pendingld
  lsq.io.rob.pendingst           := io.ooo_to_mem.lsqio.pendingst
  lsq.io.rob.commit              := io.ooo_to_mem.lsqio.commit
  lsq.io.rob.pendingPtr          := io.ooo_to_mem.lsqio.pendingPtr

//  lsq.io.rob            <> io.lsqio.rob
  lsq.io.enq            <> io.ooo_to_mem.enqLsq
  lsq.io.brqRedirect    <> redirect
  // 把从loadQueue中拿到的是否发生memoryViolation信号送出去(ctrlBlock)处理
  // ctrlBlock会基于该信号从redirectGen模块中产生flush信号
  io.mem_to_ooo.memoryViolation    <> lsq.io.rollback
  io.mem_to_ooo.lsqio.lqCanAccept  := lsq.io.lqCanAccept
  io.mem_to_ooo.lsqio.sqCanAccept  := lsq.io.sqCanAccept
  // lsq.io.uncache        <> uncache.io.lsq
  AddPipelineReg(lsq.io.uncache.req, uncache.io.lsq.req, false.B)
  AddPipelineReg(uncache.io.lsq.resp, lsq.io.uncache.resp, false.B)
  // delay dcache refill for 1 cycle for better timing
  lsq.io.refill         := delayedDcacheRefill
  lsq.io.release        := dcache.io.lsu.release
  lsq.io.lqCancelCnt <> io.mem_to_ooo.lqCancelCnt
  lsq.io.sqCancelCnt <> io.mem_to_ooo.sqCancelCnt
  // 把lsq中的Deq信号(提交个数)送到ctrlBlock
  // ctrlBlock基于此决定dispatch个数等信息
  lsq.io.lqDeq <> io.mem_to_ooo.lqDeq
  lsq.io.sqDeq <> io.mem_to_ooo.sqDeq
  lsq.io.tl_d_channel <> dcache.io.lsu.tl_d_channel

  // LSQ to store buffer
  // 从store queue写出的数据给sbuffer
  lsq.io.sbuffer        <> sbuffer.io.in
  // 从store queue给sbuffer信号, 表示store queue是否为空
  lsq.io.sqEmpty        <> sbuffer.io.sqempty
  dcache.io.force_write := lsq.io.force_write
  // Sbuffer
  sbuffer.io.csrCtrl    <> csrCtrl
  sbuffer.io.dcache     <> dcache.io.lsu.store
  sbuffer.io.memSetPattenDetected := dcache.io.memSetPattenDetected
  sbuffer.io.force_write <> lsq.io.force_write
  // flush sbuffer
  // atomicUnits执行时, 需要flush sbuffer
  val fenceFlush = io.ooo_to_mem.flushSb
  val atomicsFlush = atomicsUnit.io.flush_sbuffer.valid
  val stIsEmpty = sbuffer.io.flush.empty && uncache.io.flush.empty
  io.mem_to_ooo.sbIsEmpty := RegNext(stIsEmpty)

  // if both of them tries to flush sbuffer at the same time
  // something must have gone wrong
  // TODO: why? fence指令和atomic指令不应该同时执行, 因此不会出现同时flush的情况
  assert(!(fenceFlush && atomicsFlush))
  sbuffer.io.flush.valid := RegNext(fenceFlush || atomicsFlush)
  uncache.io.flush.valid := sbuffer.io.flush.valid

  // Vector Load/Store Queue
  vlsq.io.int2vlsu <> io.int2vlsu
  vlsq.io.vec2vlsu <> io.vec2vlsu
  vlsq.io.vlsu2vec <> io.vlsu2vec
  vlsq.io.vlsu2int <> io.vlsu2int
  vlsq.io.vlsu2ctrl <> io.vlsu2ctrl

  // AtomicsUnit: AtomicsUnit will override other control signals,
  // as atomics insts (LR/SC/AMO) will block the pipeline
  val s_normal +: s_atomics = Enum(exuParameters.StuCnt + 1)
  val state = RegInit(s_normal)

  val atomic_rs = (0 until exuParameters.StuCnt).map(exuParameters.LduCnt + _)
  val atomic_replay_port_idx = (0 until exuParameters.StuCnt)
  // 如果sta流水线中的指令有效且是AMO操作, 则生成一个st_atomics序列, 包含两个元素
  val st_atomics = Seq.tabulate(exuParameters.StuCnt)(i =>
    io.ooo_to_mem.issue(atomic_rs(i)).valid && FuType.storeIsAMO((io.ooo_to_mem.issue(atomic_rs(i)).bits.uop.ctrl.fuType))
  )

  val st_data_atomics = Seq.tabulate(exuParameters.StuCnt)(i =>
    stData(i).valid && FuType.storeIsAMO(stData(i).bits.uop.ctrl.fuType)
  )

  for (i <- 0 until exuParameters.StuCnt) when(st_atomics(i)) {
    io.ooo_to_mem.issue(atomic_rs(i)).ready := atomicsUnit.io.in.ready
    storeUnits(i).io.stin.valid := false.B

    // 如果两个sta流水线中有一个是atomic, 那么也把状态机改成s_atomics(i)
    state := s_atomics(i)
    if (exuParameters.StuCnt > 1)
      assert(!st_atomics.zipWithIndex.filterNot(_._2 == i).unzip._1.reduce(_ || _))
  }
  when (atomicsUnit.io.out.valid) {
    assert((0 until exuParameters.StuCnt).map(state === s_atomics(_)).reduce(_ || _))
    state := s_normal
  }

  // 如果有amo指令, 送入atomicsUnit处理
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
  // TODO: amo指令来源于sta, 但是走dtlb_ld进行地址翻译? 目的是什么
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
    // TODO: 为什么执行amo时不允许发出prefetch?
    loadUnits.map(i => i.io.prefetch_req.valid := false.B)

    // make sure there's no in-flight uops in load unit
    assert(!loadUnits(0).io.ldout.valid)
  }

  for (i <- 0 until exuParameters.StuCnt) when (state === s_atomics(i)) {
    atomicsUnit.io.feedbackSlow <> io.rsfeedback(atomic_rs(i)).feedbackSlow

    assert(!storeUnits(i).io.feedback_slow.valid)
  }

  // 把异常地址送出给csr, 准备异常处理
  lsq.io.exceptionAddr.isStore := io.ooo_to_mem.isStore
  // Exception address is used several cycles after flush.
  // We delay it by 10 cycles to ensure its flush safety.
  val atomicsException = RegInit(false.B)
  when (DelayN(redirect.valid, 10) && atomicsException) {
    atomicsException := false.B
  }.elsewhen (atomicsUnit.io.exceptionAddr.valid) {
    atomicsException := true.B
  }
  // TODO: 为什么atomic有异常需要处理时,不允许有新指令进入?
  val atomicsExceptionAddress = RegEnable(atomicsUnit.io.exceptionAddr.bits, atomicsUnit.io.exceptionAddr.valid)
  io.mem_to_ooo.lsqio.vaddr := RegNext(Mux(atomicsException, atomicsExceptionAddress, lsq.io.exceptionAddr.vaddr))
  XSError(atomicsException && atomicsUnit.io.in.valid, "new instruction before exception triggers\n")

  io.memInfo.sqFull := RegNext(lsq.io.sqFull)
  io.memInfo.lqFull := RegNext(lsq.io.lqFull)
  io.memInfo.dcacheMSHRFull := RegNext(dcache.io.mshrFull)

  // top-down info
  dcache.io.debugTopDown.robHeadVaddr := io.debugTopDown.robHeadVaddr
  dtlbRepeater1.io.debugTopDown.robHeadVaddr := io.debugTopDown.robHeadVaddr
  lsq.io.debugTopDown.robHeadVaddr := io.debugTopDown.robHeadVaddr
  io.debugTopDown.toCore.robHeadMissInDCache := dcache.io.debugTopDown.robHeadMissInDCache
  io.debugTopDown.toCore.robHeadTlbReplay := lsq.io.debugTopDown.robHeadTlbReplay
  io.debugTopDown.toCore.robHeadTlbMiss := lsq.io.debugTopDown.robHeadTlbMiss
  io.debugTopDown.toCore.robHeadLoadVio := lsq.io.debugTopDown.robHeadLoadVio
  io.debugTopDown.toCore.robHeadLoadMSHR := lsq.io.debugTopDown.robHeadLoadMSHR
  dcache.io.debugTopDown.robHeadOtherReplay := lsq.io.debugTopDown.robHeadOtherReplay

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
