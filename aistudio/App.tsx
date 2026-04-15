/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Sparkles, 
  User, 
  HelpCircle, 
  Zap, 
  Heart, 
  Star, 
  History, 
  CreditCard, 
  LogOut,
  ChevronRight,
  Check,
  Video,
  Edit3,
  Image as ImageIcon,
  Wand2
} from 'lucide-react';

type Tab = 'generation' | 'faq' | 'profile';
type Mode = 'generation' | 'video' | 'edit' | 'inpaint';

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('generation');
  const [activeMode, setActiveMode] = useState<Mode>('generation');
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [prompt, setPrompt] = useState('');
  
  // Settings State
  const [settings, setSettings] = useState({
    watermark: true,
    highRes: true,
    autoUpscale: true,
    autoPrompt: true,
    samplingSteps: 32,
    cfgScale: 14.5,
  });

  const toggleSetting = (key: keyof typeof settings) => {
    setSettings(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const handleSliderChange = (key: 'samplingSteps' | 'cfgScale', e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(1, x / rect.width));
    
    if (key === 'samplingSteps') {
      setSettings(prev => ({ ...prev, samplingSteps: Math.round(percentage * 50) }));
    } else {
      setSettings(prev => ({ ...prev, cfgScale: Number((percentage * 20).toFixed(1)) }));
    }
  };

  const handleGenerate = () => {
    setIsGenerating(true);
    setProgress(0);
  };

  useEffect(() => {
    if (isGenerating) {
      const interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 100) {
            clearInterval(interval);
            setTimeout(() => setIsGenerating(false), 1000);
            return 100;
          }
          return prev + 2;
        });
      }, 30);
      return () => clearInterval(interval);
    }
  }, [isGenerating]);

  const modes = [
    { id: 'generation', label: 'Create', icon: Sparkles, desc: 'Text to image' },
    { id: 'video', label: 'Motion', icon: Video, desc: 'Animate scene' },
    { id: 'edit', label: 'Refine', icon: Edit3, desc: 'Image to image' },
    { id: 'inpaint', label: 'Surgical', icon: Wand2, desc: 'Area edit' },
  ];

  return (
    <div className="min-h-screen-safe flex flex-col items-center justify-start p-6 relative overflow-hidden">
      {/* Aurora Background */}
      <div className="aurora-bg" />

      <main className="w-full max-w-md z-10 space-y-8">
        {/* Header */}
        <header className="flex justify-between items-center px-2">
          <div className="space-y-1">
            <motion.h1 
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="font-display text-4xl font-bold tracking-tight text-white flex items-center gap-3"
            >
              Angel <Heart className="text-[#ff2d95] fill-[#ff2d95] floating" size={28} />
            </motion.h1>
            <p className="text-[10px] uppercase tracking-[0.4em] text-white/50 font-black">Neural Synthesis // Elite</p>
          </div>
          <motion.div 
            whileHover={{ rotate: 180 }}
            className="w-12 h-12 rounded-2xl aurora-card flex items-center justify-center text-[#ff2d95] shadow-lg shadow-pink-500/20"
          >
            <Star size={24} fill="currentColor" />
          </motion.div>
        </header>

        {/* Tab Navigation */}
        <nav className="aurora-card p-2 flex gap-2">
          {(['generation', 'faq', 'profile'] as Tab[]).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`relative flex-1 py-3 rounded-2xl text-[10px] font-black uppercase tracking-[0.2em] transition-all duration-500 ${activeTab === tab ? 'text-white' : 'text-white/30 hover:text-white/50'}`}
            >
              {activeTab === tab && (
                <motion.div 
                  layoutId="activeTab"
                  className="absolute inset-0 bg-white/10 rounded-2xl shadow-inner"
                />
              )}
              <span className="relative z-10 flex items-center justify-center gap-2">
                {tab === 'generation' && <Sparkles size={14} />}
                {tab === 'faq' && <HelpCircle size={14} />}
                {tab === 'profile' && <User size={14} />}
                {tab}
              </span>
            </button>
          ))}
        </nav>

        {/* Content Area */}
        <AnimatePresence mode="wait">
          {activeTab === 'generation' && (
            <motion.div
              key="generation"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="space-y-6"
            >
              {/* Mode Grid */}
              <div className="grid grid-cols-2 gap-3">
                {modes.map((m) => (
                  <div
                    key={m.id}
                    onClick={() => setActiveMode(m.id as Mode)}
                    className={`mode-card ${activeMode === m.id ? 'active' : ''}`}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <m.icon size={20} className="mode-icon transition-all duration-500" />
                      {activeMode === m.id && <motion.div layoutId="check" className="w-2 h-2 rounded-full bg-[#ff2d95] shadow-[0_0_8px_#ff2d95]" />}
                    </div>
                    <div className="space-y-0.5">
                      <p className="text-xs font-bold uppercase tracking-wider">{m.label}</p>
                      <p className="text-[9px] text-white/30 font-medium">{m.desc}</p>
                    </div>
                  </div>
                ))}
              </div>

              {/* Prompt Section */}
              <div className="space-y-3">
                <div className="aurora-card p-1">
                  <textarea 
                    className="w-full h-32 bg-transparent p-5 text-sm leading-relaxed outline-none resize-none placeholder:text-white/20" 
                    placeholder="Whisper your desires to the machine..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                  />
                </div>
                <div className="flex items-center justify-between px-2">
                  <motion.button 
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => {
                      // Simulate prompt improvement
                      if (prompt) {
                        setPrompt(prev => prev + " (ultra high res, masterpiece, intricate details, cinematic lighting, 8k)");
                      }
                    }}
                    className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-all group"
                  >
                    <Wand2 size={14} className="text-[#ff2d95] group-hover:rotate-12 transition-transform" />
                    <span className="text-[10px] font-black uppercase tracking-widest text-white/70 group-hover:text-white">
                      Улучшить промт
                    </span>
                  </motion.button>
                  <span className="text-[10px] font-black text-white/20 tracking-widest">{prompt.length}/500</span>
                </div>
              </div>

              {/* Controls */}
              <div className="aurora-card p-6 space-y-8">
                <div className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex justify-between text-[10px] font-black uppercase tracking-[0.2em] text-white/40">
                      <span>Sampling Precision</span>
                      <span className="text-[#ff2d95]">{settings.samplingSteps}</span>
                    </div>
                    <div className="liquid-slider-track" onClick={(e) => handleSliderChange('samplingSteps', e)}>
                      <motion.div 
                        className="liquid-slider-fill" 
                        animate={{ width: `${(settings.samplingSteps / 50) * 100}%` }}
                      >
                        <div className="liquid-slider-thumb" />
                      </motion.div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div className="flex justify-between text-[10px] font-black uppercase tracking-[0.2em] text-white/40">
                      <span>Neural Guidance</span>
                      <span className="text-[#ff2d95]">{settings.cfgScale}</span>
                    </div>
                    <div className="liquid-slider-track" onClick={(e) => handleSliderChange('cfgScale', e)}>
                      <motion.div 
                        className="liquid-slider-fill" 
                        animate={{ width: `${(settings.cfgScale / 20) * 100}%` }}
                      >
                        <div className="liquid-slider-thumb" />
                      </motion.div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Action Button */}
              <motion.button
                onClick={handleGenerate}
                disabled={isGenerating}
                whileTap={{ scale: 0.98 }}
                className="aurora-btn w-full relative overflow-hidden group"
              >
                <AnimatePresence mode="wait">
                  {isGenerating ? (
                    <motion.div
                      key="generating"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="flex items-center justify-center gap-4"
                    >
                      <Sparkles className="animate-spin" size={20} />
                      <span>Manifesting {progress}%</span>
                    </motion.div>
                  ) : (
                    <motion.div
                      key="idle"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="flex items-center justify-center gap-4"
                    >
                      <Zap size={20} fill="currentColor" />
                      <span>Execute Synthesis</span>
                    </motion.div>
                  )}
                </AnimatePresence>
                
                {/* Progress Overlay */}
                {isGenerating && (
                  <motion.div 
                    className="absolute inset-0 bg-white/10 origin-left"
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: progress / 100 }}
                  />
                )}
              </motion.button>
            </motion.div>
          )}

          {activeTab === 'faq' && (
            <motion.div
              key="faq"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="space-y-4"
            >
              <div className="aurora-card p-8 space-y-8">
                <div className="space-y-2">
                  <h2 className="font-display text-3xl font-bold text-[#ff2d95]">Manual</h2>
                  <p className="text-[10px] uppercase tracking-[0.3em] text-white/30 font-black">System Protocols</p>
                </div>
                <div className="space-y-8">
                  {[
                    { q: "Neural Protocol", a: "Our AI uses advanced latent diffusion to manifest your desires. Guidance values increase adherence to prompt." },
                    { q: "Privacy Shield", a: "All data is processed in a secure, volatile environment. Your fantasies remain yours alone." },
                    { q: "Mastering", a: "Use Surgical mode for pixel-perfect precision. Draw over areas you wish to re-synthesize." }
                  ].map((item, i) => (
                    <div key={i} className="space-y-3 group">
                      <h3 className="text-[11px] font-black uppercase tracking-widest text-white flex items-center gap-3">
                        <div className="w-1.5 h-1.5 rounded-full bg-[#00d2ff]" /> {item.q}
                      </h3>
                      <p className="text-sm text-white/40 leading-relaxed font-medium">{item.a}</p>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'profile' && (
            <motion.div
              key="profile"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="space-y-6"
            >
              <div className="aurora-card p-10 flex flex-col items-center text-center space-y-6">
                <div className="relative">
                  <div className="w-32 h-32 rounded-[40px] aurora-card p-1 rotate-12">
                    <div className="w-full h-full rounded-[36px] bg-gradient-to-tr from-[#ff2d95] to-[#9d4dff] flex items-center justify-center -rotate-12">
                      <User size={56} className="text-white/90" />
                    </div>
                  </div>
                  <motion.div 
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ repeat: Infinity, duration: 2 }}
                    className="absolute -bottom-2 -right-2 w-10 h-10 bg-[#00d2ff] rounded-2xl flex items-center justify-center text-black shadow-lg shadow-cyan-500/40"
                  >
                    <Star size={20} fill="currentColor" />
                  </motion.div>
                </div>
                <div className="space-y-2">
                  <h2 className="font-display text-3xl font-bold">Keifer-chan</h2>
                  <div className="inline-block px-4 py-1.5 rounded-full bg-white/5 border border-white/10">
                    <p className="text-[10px] font-black uppercase tracking-[0.4em] text-[#ff2d95]">Elite Operator</p>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="aurora-card p-6 text-center">
                  <p className="text-[10px] font-black uppercase tracking-widest text-white/30 mb-2">Manifestations</p>
                  <p className="text-3xl font-bold font-display">1,248</p>
                </div>
                <div className="aurora-card p-6 text-center">
                  <p className="text-[10px] font-black uppercase tracking-widest text-white/30 mb-2">Sync Level</p>
                  <p className="text-3xl font-bold font-display text-[#00d2ff]">MAX</p>
                </div>
              </div>

              <div className="aurora-card overflow-hidden divide-y divide-white/5">
                {[
                  { icon: History, label: "History Log" },
                  { icon: CreditCard, label: "Access Tier" },
                  { icon: LogOut, label: "Terminate Session", color: "text-[#ff2d95]" }
                ].map((item, i) => (
                  <button key={i} className="w-full px-8 py-5 flex items-center justify-between hover:bg-white/5 transition-all group">
                    <div className="flex items-center gap-5">
                      <item.icon size={20} className={item.color || "text-white/30 group-hover:text-[#ff2d95]"} />
                      <span className={`text-[11px] font-black uppercase tracking-wider ${item.color || "text-white/70"}`}>{item.label}</span>
                    </div>
                    <ChevronRight size={18} className="text-white/10 group-hover:text-white/40" />
                  </button>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer Decoration */}
      <footer className="mt-12 text-center space-y-3 opacity-30">
        <div className="flex justify-center gap-6">
          <Heart size={14} className="text-[#ff2d95]" />
          <Star size={14} className="text-[#00d2ff]" />
          <Heart size={14} className="text-[#ff2d95]" />
        </div>
        <p className="text-[9px] uppercase tracking-[0.6em] font-black">Angel Arena // Neural Synthesis</p>
      </footer>
    </div>
  );
}
