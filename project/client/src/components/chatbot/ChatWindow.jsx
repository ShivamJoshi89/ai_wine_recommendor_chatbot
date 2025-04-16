// client/src/components/chatbot/ChatWindow.jsx
import React, { useState, useEffect, useRef } from 'react';
import { Box, Paper, Typography, TextField, Button } from '@mui/material';
import { motion } from 'framer-motion';
import { sendMessage } from '../../services/chatService';
import { getWineDetail } from '../../services/wineService';
import RecommendedWineCard from './RecommendedWineCard';

const ChatWindow = ({ quickReply }) => {
  const [messages, setMessages] = useState([
    { sender: 'assistant', text: "Hello! How can I help you with your wine selection today?" }
  ]);
  const [input, setInput] = useState('');
  const [error, setError] = useState('');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (quickReply) {
      setInput(quickReply);
    }
  }, [quickReply]);

  const extractWineId = (text) => {
    const regex = /\/wine-details\/([a-fA-F0-9]+)/;
    const match = text.match(regex);
    return match ? match[1] : null;
  };

  const formatAssistantText = (raw) => {
    return raw
      .replace(/\*\*/g, '')
      // eslint-disable-next-line no-useless-escape
      .replace(/\$\s*([\d]+)\.\s*([\d]+)/g, '\\$$1.$2')
      .replace(/^\s*1\./m, '\n1.')
      .replace(/^\s*2\./m, '\n2.')
      .replace(/^\s*3\./m, '\n3.')
      .replace(/^- /gm, '\n  o ')
      .replace(/\. /g, '.\n')
      .trim();
  };
  

  const handleSend = async () => {
    if (!input.trim()) return;
    const token = localStorage.getItem('token');
    if (!token) {
      setError('You must be logged in to chat.');
      return;
    }

    setMessages(prev => [...prev, { sender: 'user', text: input }]);
    setInput('');

    try {
      const data = await sendMessage(input, token);
      const raw = data.response || '';

      const wineId = extractWineId(raw);
      if (wineId) {
        const wineDetails = await getWineDetail(wineId);
        setMessages(prev => [...prev, { sender: 'assistant', wineCard: wineDetails }]);
      } else {
        const formatted = formatAssistantText(raw);
        setMessages(prev => [...prev, { sender: 'assistant', text: formatted }]);
      }
    } catch (err) {
      console.error('Error sending chat message:', err);
      setError('Failed to get a response from the assistant.');
    }
  };

  const renderMessage = (msg, idx) => {
    if (msg.sender === 'assistant' && msg.wineCard) {
      return (
        <Box key={idx} sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
          <RecommendedWineCard wine={msg.wineCard} />
        </Box>
      );
    } else {
      const isUser = msg.sender === 'user';
      return (
        <motion.div
          key={idx}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <Box
            sx={{
              display: 'flex',
              justifyContent: isUser ? 'flex-end' : 'flex-start',
              mb: 1,
            }}
          >
            <Box
              sx={{
                bgcolor: isUser ? 'primary.main' : 'background.paper',
                color: isUser ? 'primary.contrastText' : 'text.primary',
                p: 1,
                borderRadius: 2,
                maxWidth: '70%',
                whiteSpace: 'pre-line', // preserve newlines
              }}
            >
              <Typography variant="body1">
                {msg.text}
              </Typography>
            </Box>
          </Box>
        </motion.div>
      );
    }
  };

  return (
    <Paper
      sx={{
        p: 2,
        height: '70vh',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: 'transparent',
        boxShadow: 'none'
      }}
    >
      <Box sx={{ flex: 1, overflowY: 'auto', mb: 2 }}>
        {messages.map((msg, idx) => renderMessage(msg, idx))}
        <div ref={messagesEndRef} />
      </Box>

      {error && (
        <Typography variant="body2" color="error" sx={{ mb: 1 }}>
          {error}
        </Typography>
      )}

      <Box sx={{ display: 'flex' }}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => { if (e.key === 'Enter') handleSend(); }}
          sx={{
            bgcolor: 'background.paper',
            '& .MuiOutlinedInput-root': { color: 'text.primary' },
          }}
        />
        <Button
          variant="contained"
          sx={{ ml: 1, textTransform: 'uppercase' }}
          onClick={handleSend}
        >
          Send
        </Button>
      </Box>
    </Paper>
  );
};

export default ChatWindow;
