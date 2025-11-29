import React from 'react';
import { ExternalLink } from 'lucide-react';

interface VideoPlayerProps {
  url: string;
  title: string;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ url, title }) => {
  // Simple check for YouTube ID
  const getYoutubeId = (url: string) => {
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = url.match(regExp);
    return (match && match[2].length === 11) ? match[2] : null;
  };

  const videoId = getYoutubeId(url);

  if (videoId) {
    return (
      <div className="aspect-video w-full rounded-xl overflow-hidden shadow-lg border border-slate-700 bg-black">
        <iframe
          className="w-full h-full"
          src={`https://www.youtube.com/embed/${videoId}`}
          title={title}
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        />
      </div>
    );
  }

  return (
    <div className="p-6 bg-slate-800 rounded-xl flex flex-col items-center justify-center space-y-4 border border-slate-700">
      <p className="text-slate-400">Video preview not available for this URL format.</p>
      <a 
        href={url} 
        target="_blank" 
        rel="noopener noreferrer"
        className="flex items-center space-x-2 text-blue-400 hover:text-blue-300"
      >
        <span>Watch "{title}" on external site</span>
        <ExternalLink size={16} />
      </a>
    </div>
  );
};

export default VideoPlayer;