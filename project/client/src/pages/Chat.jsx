// client/src/pages/Chat.jsx
import React, { useState } from 'react';
import { Container, Typography, Box, Button, Paper } from '@mui/material';
import { motion } from 'framer-motion';
import ChatWindow from '../components/chatbot/ChatWindow';

const Chat = () => {
  const quickReplies = [
    "Show me red wines",
    "I like sparkling wine",
    "Recommend something under $30",
    "Tell me about Chardonnay"
  ];

  const [quickReplyMessage, setQuickReplyMessage] = useState('');

  const handleQuickReply = (reply) => {
    setQuickReplyMessage(reply);
  };

  return (
    <Container
      maxWidth={false}          // allow full width
      sx={{
        position: 'relative',
        zIndex: 2,
        mt: 4,
        mb: 4,
        px: { xs: 2, md: 4 },
        backgroundColor: 'transparent',
      }}
    >
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
        <Typography
          variant="h4"
          align="center"
          gutterBottom
          sx={{
            color: 'text.primary',
            textTransform: 'uppercase',
            letterSpacing: 2,
            fontWeight: 700,
          }}
        >
          Chat with Our Vino‑Sage Wine Advisor
        </Typography>
      </motion.div>

      <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap', justifyContent: 'center' }}>
        {quickReplies.map((reply, idx) => (
          <Button
            key={idx}
            variant="outlined"
            onClick={() => handleQuickReply(reply)}
            sx={{
              textTransform: 'none',
              color: 'text.primary',
              borderColor: 'text.primary'
            }}
          >
            {reply}
          </Button>
        ))}
      </Box>

      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.8, delay: 0.3 }}>
        <Box
          sx={{
            width: '100%',
            maxWidth: 1400,       // adjust as needed
            mx: 'auto',
          }}
        >
          <Paper
            elevation={0}
            sx={{
              backdropFilter: 'blur(10px)',
              backgroundColor: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.15)',
              borderRadius: 10,
              boxShadow: '0 8px 32px rgba(0,0,0,0.37)',
              p: 4,
            }}
          >
            <ChatWindow quickReply={quickReplyMessage} />
          </Paper>
        </Box>
      </motion.div>
    </Container>
  );
};

export default Chat;
