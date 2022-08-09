<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Feedback extends Model
{
    // Table Name
    protected $table = 'feedback';
    // Primary Key
    public $primary_key = 'id';
    // Timestamps
    public $timestamps = false;

    protected $fillable = [
        'id_peternakan',
        'jenis_feedback',
        'pesan',
        'waktu'
    ];

    public function peternakan()
    {
        return $this->belongsTo('App\Models\Peternakan','id_peternakan');
    }
}
