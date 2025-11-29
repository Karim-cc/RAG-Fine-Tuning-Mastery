import React from 'react';
import { Module, Lesson } from '../types';
import { BookOpen, CheckCircle, Circle, PlayCircle } from 'lucide-react';

interface SidebarProps {
  curriculum: Module[];
  currentLesson: Lesson;
  onSelectLesson: (lesson: Lesson) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ curriculum, currentLesson, onSelectLesson }) => {
  return (
    <div className="w-64 bg-slate-950 border-r border-slate-800 h-screen overflow-y-auto flex-shrink-0 hidden md:block">
      <div className="p-4 border-b border-slate-800 sticky top-0 bg-slate-950 z-10">
        <h1 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-500">
          LLM Mastery
        </h1>
        <p className="text-xs text-slate-500 mt-1">RAG & Fine-Tuning Architect</p>
      </div>

      <div className="p-2">
        {curriculum.map((module) => (
          <div key={module.id} className="mb-4">
            <h2 className="px-2 text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">
              {module.title}
            </h2>
            <div className="space-y-1">
              {module.lessons.map((lesson) => {
                const isActive = lesson.id === currentLesson.id;
                return (
                  <button
                    key={lesson.id}
                    onClick={() => onSelectLesson(lesson)}
                    className={`w-full text-left px-3 py-2 rounded-md text-sm flex items-center space-x-3 transition-colors ${
                      isActive 
                        ? 'bg-slate-800 text-white border border-slate-700' 
                        : 'text-slate-400 hover:bg-slate-900 hover:text-slate-200'
                    }`}
                  >
                    {isActive ? <PlayCircle size={16} className="text-indigo-400" /> : <Circle size={16} />}
                    <span className="truncate">{lesson.title}</span>
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Sidebar;